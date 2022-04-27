import numpy as np
import pandas as pd
import scipy.interpolate
import numpy.linalg as nl
import scipy.spatial
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def get_x0cut_inter(x0cut, n=10000):
    x0cut = x0cut[["y", "z"]].drop_duplicates()
    tck, u = scipy.interpolate.splprep([x0cut.y.values, x0cut.z.values], s=0, k=3)
    ts = np.linspace(0, 1, n)
    yzi = scipy.interpolate.splev(ts, tck)
    dfi = pd.DataFrame(np.vstack(yzi).T, columns=["y", "z"])
    dfi["x"] = 0.
    dfi["t"] = ts
    return dfi

def calc_normal(df, normal=(1, 0, 0), debug=False):
    def cross(row):
        r = np.cross(normal, (row["dx"], row["dy"], row["dz"]))
        # print(r, type(r), r / (r[0]**2 + r[1]**2 + r[2]**2)**.5)
        return r / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** .5

    curve = df.copy()
    curve["dx"] = np.gradient(curve.x)
    curve["dy"] = np.gradient(curve.y)
    curve["dz"] = np.gradient(curve.z)
    curve["ds"] = (curve.dx**2 + curve.dy**2 + curve.dz**2)**.5
    curve["s"] = 0.
    curve.iloc[1:, curve.columns.get_loc('s')] = np.cumsum((np.diff(curve.x)**2 + np.diff(curve.y)**2 + np.diff(curve.z)**2)**.5)

    n = curve.apply(cross, axis=1)
    n = np.vstack(n)
    # print(n)
    curve["nx"] = n[:, 0]
    curve["ny"] = n[:, 1]
    curve["nz"] = n[:, 2]
    if debug is True:
        return curve
    cols = df.columns.values.tolist()
    cols.extend(["s", "nx", "ny", "nz"])
    return curve[cols]

def spline_interpolation(xy, num=50, s=0.0, k=1, nest=-1):
    """Return interpolated polyline

    :param num int: number of points
    :param s float: smootheness
    :param k int: spline order
    :param nest int: estimate of number of knots needed
    :returns: interpolated polyline
    :rtype: Polyline2D

    """
    tckp, u = scipy.interpolate.splprep(xy.T, s=s, k=k, nest=nest)
    x, y = scipy.interpolate.splev(np.linspace(0, 1, num=num), tckp)
    return np.c_[x, y]


def detect_side(xy_source, xy_target, color1="r", color2="g", reverse=False, k=3):
    """calculate side of target points c=1|-1"""
    refpoints = spline_interpolation(xy_target, num=int(1e5), s=0.0, k=k, nest=-1)
    refpointstangent = spline_interpolation(xy_target, num=int(1e5), s=0.0, k=k, nest=-1)
    refpointstangent = np.vstack((refpointstangent, refpointstangent[-1]))
    reftree = scipy.spatial.cKDTree(refpoints)
    distances, indices = reftree.query(xy_source)
    nidindices = np.array(list(range(len(xy_source))))
    oldpoints = xy_source[nidindices]
    newpoints = refpoints[indices]
    newpointstangent = refpointstangent[indices]
    newpointstangent2 = refpointstangent[indices + 1]
    tvec = newpointstangent - newpointstangent2
    nvec = newpoints - oldpoints
    plinesource_pkts = xy_source
    cs = []
    for i, nid in enumerate(nidindices):
        pkt = plinesource_pkts[nid]

        t = tvec[i] / nl.norm(tvec[i])
        n = nvec[i] / nl.norm(nvec[i])
        r = np.cross((0, 0, 1), (tvec[i][0], tvec[i][1], 0))[:2]
        r = r / nl.norm(r)
        c = np.cross(t, n)  # * np.sign(a)
        if np.isnan(c):
            c = cs[-1][2]
        cs.append([plinesource_pkts[i][0], plinesource_pkts[i][1], np.sign(c), newpoints[i][0], newpoints[i][1]])

    def get_color(c, alpha=.9):
        if c > 0:
            return color1
        else:
            return color2

    dfs = pd.DataFrame(cs, columns=["x", "y", "c", "x_t", "y_t"])
    dfs["color"] = dfs.c.apply(get_color)
    if reverse is True:
        dfs["c"] *= -1
    return dfs


def get_x0cut_deviation(x0cut, x0cutref, n=1000):
    """get_x0cut_deviation(x0cut, x0cutref, n=1000)

    """
    x0cuti = get_x0cut_inter(x0cut, n=10000)
    x0cutrefi = get_x0cut_inter(x0cutref, n=n)

    tree = scipy.spatial.KDTree(x0cuti[["y", "z"]])
    distances, target_ids = tree.query(x0cutrefi[["y", "z"]])
    x0cutrefi["deviation"] = distances
    x0cutrefi["target_idx"] = target_ids
    x0cutrefj = pd.merge(x0cutrefi,
                         x0cuti.iloc[target_ids].reset_index().rename(
                             columns={"x": "x_d", "y": "y_d", "z": "z_d", "index": "target_idx", "t": "t_d"}),
                         left_on="target_idx", right_on="target_idx")
    dfs = detect_side(x0cutrefj[["y", "z"]].drop_duplicates().values,
                      x0cutrefj[["y_d", "z_d"]].drop_duplicates().values,
                      reverse=True)
    x0cutrefk = x0cutrefj.merge(dfs[["c", "color"]], left_index=True, right_index=True)
    x0cutrefk["deviationc"] = x0cutrefk["deviation"] * x0cutrefk["c"]
    return x0cutrefk


def get_x0cut_abs_deviation(x0cut, x0cutref, n=1000):
    """get_x0cut_abs_deviation(x0cut, x0cutref, n=1000)

    """
    x0cuti = get_x0cut_inter(x0cut, n=10000)
    x0cutrefi = get_x0cut_inter(x0cutref, n=n)

    tree = scipy.spatial.KDTree(x0cuti[["y", "z"]])
    distances, target_ids = tree.query(x0cutrefi[["y", "z"]])
    x0cutrefi["deviation"] = distances
    x0cutrefi["target_idx"] = target_ids
    x0cutrefj = pd.merge(x0cutrefi,
                         x0cuti.iloc[target_ids].reset_index().rename(
                             columns={"x": "x_d", "y": "y_d", "z": "z_d", "index": "target_idx", "t": "t_d"}),
                         left_on="target_idx", right_on="target_idx")
    return x0cutrefj


def add_parts(designparam, zt_x0cut, r=4.5, color_lu="r", color_lm="b",
              color_t="g",
              color_rm="m",
              color_ru="cyan"
              ):
    zt_x0cut.loc[:, "color"] = "k"  # "(.2,.3,.3,.9)"
    zt = designparam.Ziehtiefe

    zt_x0cut.loc[:, "Ziehtiefe"] = zt
    zt_x0cut.loc[(zt_x0cut.y <= 0) & (zt_x0cut.z <= r), "color"] = color_lu
    zt_x0cut.loc[(zt_x0cut.y > 0) & (zt_x0cut.z <= r), "color"] = color_ru
    zt_x0cut.loc[(zt_x0cut.z >= (zt - r)), "color"] = color_t
    zt_x0cut.loc[(zt_x0cut.y <= 0) & (zt_x0cut.z > r) & (zt_x0cut.z < (zt - r)), "color"] = color_lm
    zt_x0cut.loc[(zt_x0cut.y > 0) & (zt_x0cut.z > r) & (zt_x0cut.z < (zt - r)), "color"] = color_rm

    zt_x0cut.loc[:, "part"] = "?"  # "(.2,.3,.3,.9)"
    zt_x0cut.loc[(zt_x0cut.y <= 0) & (zt_x0cut.z <= r), "part"] = "lu"
    zt_x0cut.loc[(zt_x0cut.y > 0) & (zt_x0cut.z <= r), "part"] = "ru"
    zt_x0cut.loc[(zt_x0cut.z >= (zt - r)), "part"] = "to"
    zt_x0cut.loc[(zt_x0cut.y <= 0) & (zt_x0cut.z > r) & (zt_x0cut.z < (zt - r)), "part"] = "lm"
    zt_x0cut.loc[(zt_x0cut.y > 0) & (zt_x0cut.z > r) & (zt_x0cut.z < (zt - r)), "part"] = "rm"

    zt_x0cut.loc[zt_x0cut.part == "lu", "tp"] = (
            zt_x0cut.loc[zt_x0cut.part == "lu", "t"] - zt_x0cut.loc[
        zt_x0cut.part == "lu", "t"].min())  # + 1./len(zt_x0cut.loc[zt_x0cut.part == "lu", "t"])
    zt_x0cut.loc[zt_x0cut.part == "lu", "tp"] = zt_x0cut.loc[zt_x0cut.part == "lu", "t"] / zt_x0cut.loc[
        zt_x0cut.part == "lu", "t"].max()
    zt_x0cut.loc[zt_x0cut.part == "lm", "tp"] = (
            zt_x0cut.loc[zt_x0cut.part == "lm", "t"] - zt_x0cut.loc[zt_x0cut.part == "lm", "t"].min())
    zt_x0cut.loc[zt_x0cut.part == "lm", "tp"] = zt_x0cut.loc[zt_x0cut.part == "lm", "tp"] + (
            zt_x0cut.loc[zt_x0cut.part == "lm", "tp"].max() - zt_x0cut.loc[zt_x0cut.part == "lm", "tp"].min()) / (
                                                        len(zt_x0cut.loc[zt_x0cut.part == "lm", "tp"]) + 0)
    zt_x0cut.loc[zt_x0cut.part == "lm", "tp"] = (zt_x0cut.loc[zt_x0cut.part == "lm", "tp"]) / zt_x0cut.loc[
        zt_x0cut.part == "lm", "tp"].max() + 1

    zt_x0cut.loc[zt_x0cut.part == "to", "tp"] = (
            zt_x0cut.loc[zt_x0cut.part == "to", "t"] - zt_x0cut.loc[
        zt_x0cut.part == "to", "t"].min())
    zt_x0cut.loc[zt_x0cut.part == "to", "tp"] = zt_x0cut.loc[zt_x0cut.part == "to", "tp"] + (
            zt_x0cut.loc[zt_x0cut.part == "to", "tp"].max() - zt_x0cut.loc[zt_x0cut.part == "to", "tp"].min()) / (
                                                        len(zt_x0cut.loc[zt_x0cut.part == "to", "tp"]) + 0)
    zt_x0cut.loc[zt_x0cut.part == "to", "tp"] = (zt_x0cut.loc[zt_x0cut.part == "to", "tp"]) / zt_x0cut.loc[
        zt_x0cut.part == "to", "tp"].max() + 2

    zt_x0cut.loc[zt_x0cut.part == "rm", "tp"] = (
            zt_x0cut.loc[zt_x0cut.part == "rm", "t"] - zt_x0cut.loc[
        zt_x0cut.part == "rm", "t"].min())
    zt_x0cut.loc[zt_x0cut.part == "rm", "tp"] = zt_x0cut.loc[zt_x0cut.part == "rm", "tp"] + (
            zt_x0cut.loc[zt_x0cut.part == "rm", "tp"].max() - zt_x0cut.loc[zt_x0cut.part == "rm", "tp"].min()) / (
                                                        len(zt_x0cut.loc[zt_x0cut.part == "rm", "tp"]) + 0)
    zt_x0cut.loc[zt_x0cut.part == "rm", "tp"] = (zt_x0cut.loc[zt_x0cut.part == "rm", "tp"]) / zt_x0cut.loc[
        zt_x0cut.part == "rm", "tp"].max() + 3

    zt_x0cut.loc[zt_x0cut.part == "ru", "tp"] = (
            zt_x0cut.loc[zt_x0cut.part == "ru", "t"] - zt_x0cut.loc[
        zt_x0cut.part == "ru", "t"].min())  # + 1./len(zt_x0cut.loc[zt_x0cut.part == "ru", "t"])
    zt_x0cut.loc[zt_x0cut.part == "ru", "tp"] = zt_x0cut.loc[zt_x0cut.part == "ru", "tp"] + (
            zt_x0cut.loc[zt_x0cut.part == "ru", "tp"].max() - zt_x0cut.loc[zt_x0cut.part == "ru", "tp"].min()) / (
                                                        len(zt_x0cut.loc[zt_x0cut.part == "ru", "tp"]) + 0)
    zt_x0cut.loc[zt_x0cut.part == "ru", "tp"] = (zt_x0cut.loc[zt_x0cut.part == "ru", "tp"]) / zt_x0cut.loc[
        zt_x0cut.part == "ru", "tp"].max() + 4
    return zt_x0cut


def zt2ref(df, zt_source, zt_target=50, rb=5, rt=3):
    """df[x y z], source zt

    zt .. actual Ziehtiefe
    zt_ref .. target Ziehziefe (default=50)
    """

    dfn = df.copy()
    # zref = df.zmax()
    zs = [[-rb, -rb], [0, 0], [rb, rb], [zt_source - rt, zt_target - rt], [zt_source, zt_target], [zt_source + 100, zt_target + 100]]
    zs = pd.DataFrame(zs, columns=["z", "z_s"])
    znew = np.interp(df.z, zs.z, zs.z_s)
    dfn.loc[:, "z"] = znew
    return dfn


class BananaKNeighborsRegressor():
    def __init__(self, inputparameter, n_neighbors=20, weights="distance"):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        self.scaler = StandardScaler()
        self.inputparameter = inputparameter
        self.outputparameter = None

    def fit(self, X, y):
        self.outputparameter = y.name
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X[self.inputparameter]), columns=X[self.inputparameter].columns)
        self.model.fit(X_scaled, y)

    def predict(self, T):
        T = T[self.inputparameter]
        T_scaled = pd.DataFrame(self.scaler.transform(T), columns=T.columns)
        T[self.outputparameter] = self.model.predict(T_scaled)
        return T


class BananaSVR():
    def __init__(self, inputparameter, C=100, gamma=0.1, epsilon=0.1, kernel="rbf"):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler = StandardScaler()
        self.inputparameter = inputparameter
        self.outputparameter = None

    def fit(self, X, y):
        self.outputparameter = y.name
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X[self.inputparameter]), columns=X[self.inputparameter].columns)
        self.model.fit(X_scaled, y)

    def predict(self, T):
        T = T[self.inputparameter]
        T_scaled = pd.DataFrame(self.scaler.transform(T), columns=T.columns)
        T[self.outputparameter] = self.model.predict(T_scaled)
        return T

def gen_grid_data(x, y):
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
    df = pd.DataFrame({x.name: x_grid.ravel(), y.name: y_grid.ravel()})
    return df

import dynapy
import os

def get_cuts(design, doe, root_dest):
    design_id = int(design.replace("Design", ""))
    design_doe = doe.loc[design_id]
    nodes_filepath = os.path.join(root_dest, design, "nodes_thickness.csv.gz")
    nodes = pd.read_csv(nodes_filepath, index_col=0)
    nodes["y0"] = nodes.y
    nodes["y"] += 500
    nodes = dynapy.geom.cylindrical_from_cartesian(nodes)

    contour_filepath = os.path.join(root_dest, design, f"{design}_contour_50.csv.gz")
    contour = pd.read_csv(contour_filepath, index_col=0)
    contour["y0"] = contour.y
    contour["y"] += 500
    contour = dynapy.geom.cylindrical_from_cartesian(contour)
    nodes_flange = nodes[(nodes.c_rho < 450) & (nodes.c_rho > 400)]
    nodes_web = nodes[(nodes.c_rho > 420) & (nodes.c_rho < 490) & (nodes.z > 10) & (nodes.z < 40)]

    n_x = 100
    n_y = 100

    x = pd.Series(np.deg2rad(np.linspace(70, 90, n_x)), name="c_phi")
    y = pd.Series(np.linspace(430, 430, n_y), name="c_rho")
    # df_flange_cut = banana_lib.gen_grid_data(x, y)
    df_web_cut = pd.DataFrame({"c_phi": x})
    df_web_cut["c_z"] = 25.
    df_web_cut.head()

    inputparameter = ["c_phi", "c_z"]
    X = nodes_web  # [inputparameter]
    y = nodes_web.c_rho
    webinterpol = BananaKNeighborsRegressor(inputparameter, n_neighbors=10, weights="distance")
    # webinterpol = banana_lib.BananaSVR(inputparameter, C=100, epsilon=0.1)
    webinterpol.fit(X, y)

    df_web_cut = webinterpol.predict(T=df_web_cut[inputparameter])
    df_web_cut

    x = pd.Series(np.deg2rad(np.linspace(70, 90, n_x)), name="c_phi")
    y = pd.Series(np.linspace(430, 430, n_y), name="c_rho")
    # df_flange_cut = banana_lib.gen_grid_data(x, y)
    df_flange_cut = pd.DataFrame({"c_phi": x})
    df_flange_cut["c_rho"] = 430.
    df_flange_cut.head()

    inputparameter = ["c_phi", "c_rho"]
    X = nodes_flange  # [inputparameter]
    y = nodes_flange.c_z
    webinterpol = BananaKNeighborsRegressor(inputparameter, n_neighbors=7, weights="distance")
    webinterpol = BananaSVR(inputparameter, C=1, gamma=0.1, epsilon=0.1)
    webinterpol.fit(X, y)

    df_flange_cut = webinterpol.predict(T=df_flange_cut[inputparameter])
    return {"df_flange_cut":df_flange_cut, "df_web_cut":df_web_cut}