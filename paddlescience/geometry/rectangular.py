# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .geometry_discrete import GeometryDiscrete
from .geometry import Geometry
import numpy as np
import vtk
import matplotlib.pyplot as plt
import math
import pdb


# Rectangular
class Rectangular(Geometry):
    """
    Two dimentional rectangular

    Parameters:
        space_origin: cordinate of left-bottom point of rectangular
        space_extent: extent of rectangular

    Example:
        >>> import paddlescience as psci
        >>> geo = psci.geometry.Rectangular(space_origin=(0.0,0.0), space_extent=(1.0,1.0))

    """

    # init function
    def __init__(self, space_origin=None, space_extent=None):
        super(Rectangular, self).__init__(False, None, None, space_origin,
                                          space_extent)

        # check inputs and set dimension
        self.space_origin = (space_origin, ) if (
            np.isscalar(space_origin)) else space_origin
        self.space_extent = (space_extent, ) if (
            np.isscalar(space_extent)) else space_extent

        lso = len(self.space_origin)
        lse = len(self.space_extent)
        self.space_ndims = lso
        if (lso != lse):
            print(
                "ERROR: Please check dimention of space_origin and space_extent."
            )
            exit()
        elif lso == 1:
            self.space_shape = "rectangular_1d"
        elif lso == 2:
            self.space_shape = "rectangular_2d"
        elif lso == 3:
            self.space_shape = "rectangular_3d"
        else:
            print("ERROR: Rectangular supported is should be 1d/2d/3d.")

    # domain discretize
    def discretize(self, time_nsteps=None, space_nsteps=None):

        # check input
        self.space_nsteps = (space_nsteps, ) if (
            np.isscalar(space_nsteps)) else space_nsteps

        # discretization time space with linspace
        steps = []
        if self.time_dependent == True:
            time_steps = np.linspace(
                self.time_origin, self.time_extent, time_nsteps, endpoint=True)

        # discretization each space dimention with linspace
        for i in range(self.space_ndims):
            steps.append(
                np.linspace(
                    self.space_origin[i],
                    self.space_extent[i],
                    self.space_nsteps[i],
                    endpoint=True))

        # meshgrid and stack to cordinates
        if self.time_dependent == True:
            nsteps = time_nsteps
            ndims = self.space_ndims + 1
        else:
            nsteps = 1
            ndims = self.space_ndims
        for i in self.space_nsteps:
            nsteps *= i

        if (self.space_ndims == 1):
            domain = steps[0]
        if (self.space_ndims == 2):
            mesh = np.meshgrid(steps[1], steps[0], sparse=False, indexing='ij')
            domain = np.stack(
                (mesh[1].reshape(-1), mesh[0].reshape(-1)), axis=-1)
        elif (self.space_ndims == 3):
            mesh = np.meshgrid(
                steps[2], steps[1], steps[0], sparse=False, indexing='ij')
            domain = np.stack(
                (mesh[2].reshape(-1), mesh[1].reshape(-1),
                 mesh[0].reshape(-1)),
                axis=-1)

        # 补充中间圆柱边界
        bc_inner_index = []
        id_domain = 0
        id_inner = 0
        theta = 0
        x_circle = []
        y_circle = []
        theta = 0
        r = 0.1
        while theta <= 2 * math.pi:
            x_circle = np.append(x_circle, r * math.cos(theta))
            y_circle = np.append(y_circle, r * math.sin(theta))
            theta += 0.01
        x_c1 = x_circle.reshape(len(x_circle), 1)
        y_c1 = y_circle.reshape(len(y_circle), 1)
        xy_circle = np.hstack([x_c1, y_c1])

        # 删除domain内部的点
        index_o = 0
        index_i = 0
        bc_inner = []
        for point in domain:
            for inner_p in xy_circle:
                if abs(inner_p[0]) - abs(point[0]) > 0.02 and abs(inner_p[1]) - abs(point[1]) > 0.02:
                    index_i = index_o
                    bc_inner.append(index_i)
            index_o += 1
        domain = np.delete(domain, np.unique(bc_inner), axis=0)

        # 定义完整bc_index:
            # 1) 定义圆柱周边边界 
        nbc = 0
        id_o = 0
        index_c = 0
        bc_cir = []
        for pp in domain:
            for bp in xy_circle:
                if abs(bp[0] - pp[0]) <= 0.02 and abs(bp[1] - pp[1]) <= 0.02:
                    index_c = id_o
                    bc_cir.append(index_c)
            id_o += 1
        bc_cir = np.unique(bc_cir)
        nbc += len(bc_cir) # 记录当前bc_index中包含的元素数量,代表圆柱边界的点数量

            # 2) 定义外围边界
        
        id_r = 0
        index_b = 0
        bc_side = []
        for pp in domain:
            if abs(self.space_origin[0] - pp[0]) < 0.0001 or abs(self.space_extent[0] - pp[0]) < 0.0001 or abs(self.space_origin[1] - pp[1]) < 0.0001 or abs(self.space_extent[1] - pp[1]) < 0.0001:
                index_b = id_r
                bc_side.append(index_b)
            id_r += 1
        bc_side = np.unique(bc_side)
        
        # 定义边界点总数
        nbc += len(bc_side)

        # 定义bc_index,指的是domain中点的id数组
        base_bc_index = np.vstack((bc_side.reshape(-1, 1), bc_cir.reshape(-1, 1)))
        
        bc_index = np.ndarray(nbc, dtype=int)
        for i in range(len(base_bc_index)):
            bc_index[i] = base_bc_index[i][0]

        """
        # 得到圆柱外边界的节点index
        for point_domain in domain:
            id_domain += 1
            for point_circle in xy_circle:
                if abs(point_domain[0]) < abs(point_circle[0])  and abs(point_domain[1])<abs(point_circle[1]):
                    id_inner = id_domain
                    bc_inner_index.append(id_inner)
                else:
                    continue
        pdb.set_trace()
        bc_inter_index = np.unique(bc_inner_index)
        # 基于外部边界以及内部圆柱边界共同定义bc_index
        # bc_index TODO optimize

 
        if (self.space_ndims == 1):
            bc_index = np.ndarray(2, dtype=int)
            bc_index[0] = 0
            bc_index[1] = self.space_nsteps[-1]
        elif (self.space_ndims == 2):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nbc = nx * ny - (nx - 2) * (ny - 2)
            bc_index = np.ndarray(nbc + len(bc_inter_index), dtype=int)
            nbc = 0
            for i in range(len(bc_inter_index)):
                bc_index[i] = bc_inter_index[i]
                nbc +=1 
            for j in range(ny):
                for i in range(nx):
                    if (j == 0 or j == ny - 1 or i == 0 or i == nx - 1):
                        bc_index[nbc] = j * nx + i
                        nbc += 1
        elif (self.space_ndims == 3):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            nbc = nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
            bc_index = np.ndarray(nbc, dtype=int)
            nbc = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if (k == 0 or k == nz - 1 or j == 0 or j == ny - 1 or
                                i == 0 or i == nx - 1):
                            bc_index[nbc] = k * nx * ny + j * nx + i
                            nbc += 1

        #print(domain)
        #print(bc_index)
        """

        # IC index TODO

        # return discrete geometry
        geo_disc = GeometryDiscrete()
        if self.time_dependent == True:
            geo_disc.set_time_nsteps(time_nsteps)
            geo_disc.set_time_steps(time_steps)
        geo_disc.set_domain(
            space_domain=domain,
            space_origin=self.space_origin,
            space_extent=self.space_extent)
        geo_disc.set_bc_index(bc_index)

        vtk_obj_name, vtk_obj, vtk_data_size = self.obj_vtk()
        geo_disc.set_vtk_obj(vtk_obj_name, vtk_obj, vtk_data_size)

        # mpl_obj, mpl_data_shape = self.obj_mpl()
        # geo_disc.set_mpl_obj(mpl_obj, mpl_data_shape)

        return geo_disc
    #########################
    #vtk还得修改,对应圆柱绕流,需要重新定义points,TODO
    # visu vtk
    def obj_vtk(self):
        # prepare plane obj 2d
        if self.space_ndims == 2:
            vtkobjname = "vtkPlanceSource"
            self.plane = vtk.vtkPlaneSource()
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            self.plane.SetResolution(nx - 1, ny - 1)
            self.plane.SetOrigin(
                [self.space_origin[0], self.space_origin[1], 0])
            self.plane.SetPoint1(
                [self.space_extent[0], self.space_origin[1], 0])
            self.plane.SetPoint2(
                [self.space_origin[0], self.space_extent[1], 0])
            self.plane.Update()
            vtk_data_size = self.plane.GetOutput().GetNumberOfPoints()
            return vtkobjname, self.plane, vtk_data_size
        elif self.space_ndims == 3:
            vtkobjname = "vtkImageData"
            self.img = vtk.vtkImageData()
            self.img.SetOrigin(self.space_origin[0], self.space_origin[1],
                               self.space_origin[2])
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            self.img.SetDimensions(nx, ny, nz)
            vtk_data_size = self.img.GetNumberOfPoints()
            return vtkobjname, self.img, vtk_data_size


# # visu matplotlib
# def obj_mpl(self):
#     # prepare plan obj 2d
#     if self.space_ndims == 2:
#         fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
#     return self.ax, (self.space_nsteps[0], self.space_nsteps[1])
