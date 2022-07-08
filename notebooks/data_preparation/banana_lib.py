import numpy as np
import pandas as pd
import scipy.interpolate
import numpy.linalg as nl
import scipy.spatial
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

import os


def get_x0cut_inter(x0cut, n=10000):
    x0cut = x0cut[["y", "z"]].drop_duplicates()
    tck, u = scipy.interpolate.splprep([x0cut.y.values, x0cut.z.values], s=0, k=3)
    ts = np.linspace(0, 1, n)
    yzi = scipy.interpolate.splev(ts, tck)
    dfi = pd.DataFrame(np.vstack(yzi).T, columns=["y", "z"])
    dfi["x"] = 0.
    dfi["t"] = ts
    return dfi


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
    zs = [[-rb, -rb], [0, 0], [rb, rb], [zt_source - rt, zt_target - rt], [zt_source, zt_target],
          [zt_source + 100, zt_target + 100]]
    zs = pd.DataFrame(zs, columns=["z", "z_s"])
    znew = np.interp(df.z, zs.z, zs.z_s)
    dfn.loc[:, "z"] = znew
    return dfn


class Model:
    def __init__(self, design, h5filepath, fit=False):
        self.parameter = {}
        self.design = design
        self.load_h5(h5filepath)
        self.outputparameter = self.collect_outputparameter()

        if fit is True:
            self.fit()

    def collect_outputparameter(self):
        outputparameter = []
        for col in self.nodes.columns:
            if col in ["nid", "nid0", "u", "v", "y0", "c_z"]:
                continue
            else:
                outputparameter.append(col)
        # self.outputparameter = outputparameter
        return outputparameter

    def fit(self, outputparameter=None):
        if outputparameter is None:
            outputparameter = self.collect_outputparameter()
        self.XYZ = XYZBananaKNeighborsRegressor(["u", "v"], outputparameter)
        self.XYZ.fit(self.nodes)

    def fix_z(self):
        m = self
        zm = m.nodes[(abs(m.nodes.x) < 10) & (abs(m.nodes.y) < 10)].z.mean()
        zr = m.nodes[(abs(m.nodes.x) < 10) & (abs(m.nodes.y + 100) < 10)].z.mean()
        if zm < zr:
            print("flipz")
            m.nodes.z *= -1
            m.nodes.z -= m.nodes.z.min()
            m.nodes.c_z *= -1
            m.nodes.t *= -1
        return self.nodes

    def get_thinning(self, thickness0):
        if "thickness" in self.nodes and isinstance(thickness0, float):
            self.nodes["thinning"] = self.nodes["thickness"] / thickness0

    def get_deviation_uv(self, ref_model):
        m = self
        m.nodes0 = ref_model.XYZ.predict(m.nodes)
        m.nodes0["deviation_uv_x"] = m.nodes0.x - m.nodes.x
        m.nodes0["deviation_uv_y"] = m.nodes0.y - m.nodes.y
        m.nodes0["deviation_uv_z"] = m.nodes0.z - m.nodes.z
        m.nodes0["deviation_uv"] = (m.nodes0.deviation_uv_x ** 2 +
                                    m.nodes0.deviation_uv_y ** 2 +
                                    m.nodes0.deviation_uv_z ** 2) ** .5
        m.nodes["deviation_uv_x"] = m.nodes0["deviation_uv_x"]
        m.nodes["deviation_uv_y"] = m.nodes0["deviation_uv_y"]
        m.nodes["deviation_uv_z"] = m.nodes0["deviation_uv_z"]
        m.nodes["deviation_uv"] = m.nodes0["deviation_uv"]
        return m.nodes0

    def plot_3d(self, filepath=None):
        m = self
        fig = plt.figure(1, figsize=(18, 6))
        fig.clf()
        ax = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122)
        ax.plot_trisurf(m.nodes.x, m.nodes.y, m.nodes.z, triangles=m.simplices, cmap=plt.cm.Spectral_r,
                        alpha=.3, shade=False)
        ax2.tripcolor(m.nodes.x, m.nodes.y, m.simplices, m.nodes.z, alpha=.3, edgecolor="k", cmap=plt.cm.Spectral_r)
        for u in np.linspace(0., 1, 31):
            df_x0cut = pd.DataFrame({"u": u, "v": np.linspace(0, 1, 500)})
            xyzi = m.XYZ.predict(df_x0cut[["u", "v"]])
            ax.plot(xyzi.x, xyzi.y, xyzi.z, c="k", lw=.5)
            ax2.plot(xyzi.x, xyzi.y, c="k", lw=.5)
        for v in np.linspace(0., 1, 41):
            df_x0cut = pd.DataFrame({"u": np.linspace(0, 1, 500), "v": v})
            xyzi = m.XYZ.predict(df_x0cut[["u", "v"]])
            ax.plot(xyzi.x, xyzi.y, xyzi.z, c="k", lw=.5)
            ax2.plot(xyzi.x, xyzi.y, c="k", lw=.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-220, 220)
        ax.set_ylim(-130, 130)
        ax.set_zlim(0, 80)
        ax2.set_xlim(-220, 220)
        ax2.set_ylim(-130, 130)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")
        ax2.set_title(f"Design{self.design:04d}")
        fig.tight_layout()
        if filepath is not None:
            if not filepath.endswith(".png"):
                filepath = f"{filepath}.png"
            fig.savefig(filepath)
        return fig

    def plot_uv_result(self, result_name, ax=None):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig = ax.get_figure()
        m = self
        tpc = ax.tripcolor(m.nodes.u, m.nodes.v, m.simplices, m.nodes[result_name], alpha=1, edgecolor="w",
                           shading='gouraud')
        cbar = fig.colorbar(tpc, fraction=.05, label=result_name, ax=ax)
        for u in np.linspace(0., 1, 31):
            df_x0cut = pd.DataFrame({"u": u, "v": np.linspace(0, 1, 500)})
            xyzi = m.XYZ.predict(df_x0cut[["u", "v"]])
            ax.plot(xyzi.u, xyzi.v, c="k", lw=.5, alpha=.1)
        for v in np.linspace(0., 1, 41):
            df_x0cut = pd.DataFrame({"u": np.linspace(0, 1, 500), "v": v})
            xyzi = m.XYZ.predict(df_x0cut[["u", "v"]])
            ax.plot(xyzi.u, xyzi.v, c="k", lw=.5, alpha=.1)
        return ax

    def load_h5(self, h5filepath):
        with pd.HDFStore(h5filepath, mode="r") as store:
            #            print(store.keys(), "/parameter" in store.keys())
            self.simplices = store.get(f'/simplices')
            self.nodes = store.get(f'/nodes')
            self.elements = store.get(f'/elements')
            try:
                for col in ["tc", "rc", "y0"]:
                    self.nodes.drop(columns=[col], inplace=True)
            except:
                pass
            if "/parameter" in store.keys():
                self.parameter = store.get(f'/parameter').to_dict()
            else:
                self.parameter = {}

    def save_h5(self, h5filepath):
        with pd.HDFStore(h5filepath, mode="w") as store:
            store.put(f'/simplices', self.simplices)
            store.put(f'/nodes', self.nodes)
            store.put(f'/elements', self.elements)
            store.put(f'/parameter', pd.Series(self.parameter))

    def get_contour_df(self, nn=3500):
        m = self
        df_top = pd.DataFrame({"u": np.linspace(0, 1, nn), "v": 1})
        df_right = pd.DataFrame({"u": 1., "v": np.linspace(1, 0, nn)})
        df_bot = pd.DataFrame({"u": np.linspace(1, 0, nn), "v": 0})
        df_left = pd.DataFrame({"u": 0., "v": np.linspace(0, 1, nn)})
        df_bot = m.XYZ.predict(df_bot)
        df_top = m.XYZ.predict(df_top)
        df_left = m.XYZ.predict(df_left)
        df_right = m.XYZ.predict(df_right)
        df_contour = pd.concat([df_top, df_right, df_bot, df_left])
        return df_contour

    def copy(self):
        import copy
        return copy.deepcopy(self)


class BananaKNeighborsRegressor():
    def __init__(self, inputparameter, n_neighbors=20, weights="distance"):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        self.scaler = StandardScaler()
        self.inputparameter = inputparameter
        self.outputparameter = None

    def fit(self, X, y):
        self.outputparameter = y.name
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X[self.inputparameter]),
                                columns=X[self.inputparameter].columns)
        self.model.fit(X_scaled, y)

    def predict(self, T):
        T = T[self.inputparameter]
        T_scaled = pd.DataFrame(self.scaler.transform(T), columns=T.columns)
        T[self.outputparameter] = self.model.predict(T_scaled)
        return T


class XYZBananaKNeighborsRegressor:
    def __init__(self, inputparameter, outputparameter, n_neighbors=13, weights="distance"):
        self.inputparameter = inputparameter
        self.outputparameter = outputparameter
        self.models = {}
        for o in self.outputparameter:
            self.models[o] = BananaKNeighborsRegressor(inputparameter, n_neighbors=13, weights="distance")

    def fit(self, X):
        for o in self.outputparameter:
            self.models[o].fit(X, X[o])

    def predict(self, df):
        T = df[self.inputparameter]
        ps = [T]
        for o in self.outputparameter:
            p = self.models[o].predict(T=T)
            ps.append(p[o])
        xyzi = pd.concat(ps, axis=1)
        return xyzi


class BananaSVR():
    def __init__(self, inputparameter, C=100, gamma=0.1, epsilon=0.1, kernel="rbf"):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler = StandardScaler()
        self.inputparameter = inputparameter
        self.outputparameter = None

    def fit(self, X, y):
        self.outputparameter = y.name
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X[self.inputparameter]),
                                columns=X[self.inputparameter].columns)
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

