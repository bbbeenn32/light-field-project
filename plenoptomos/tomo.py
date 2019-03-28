#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:30:00 2017

@author: vigano

This module contains the tomographic refocusing routines.
"""

import numpy as np

import time as tm

import astra

from . import lightfield
from . import solvers
from . import psf


class Projector(object):
    """Projector class: it allows to forward-project and back-project light-fields.

    This class should not be used directly, but rather through the functions
    provided in the containing module.
    """

    def __init__(self, camera : lightfield.Camera, zs, flat=None, shifts_vu=(None, None), \
                 beam_geometry='cone', domain='object', psf_d=None, \
                 use_otf=True, mode='independent', up_sampling=1, vu_origin=None, \
                 super_sampling=1, gpu_index=-1):
        self.mode = mode # can be: {'independent' | 'simultaneous' | 'range'}
        self.up_sampling = up_sampling
        self.super_sampling = super_sampling
        self.gpu_index = gpu_index
        self.beam_geometry = beam_geometry
        self.domain = domain
        self.vu_origin = vu_origin

        self.projectors = []
        self.Ws = []

        self.camera = camera
        self.img_size = np.concatenate((camera.data_size_vu, camera.data_size_ts))
        self.img_size_us = np.concatenate((camera.data_size_vu, np.array(camera.data_size_ts) * self.up_sampling))
        self.photo_size_2D = np.array(camera.data_size_vu) * np.array(camera.data_size_ts)

        self.flat = flat
        if self.flat is not None:
            self.flat = np.reshape(flat, np.concatenate(((-1, ), self.img_size)))
        self.shifts_vu = shifts_vu

        self._initialize_geometry(zs)

        if psf_d is None:
            self.psf = ()
        elif isinstance(psf_d, str) and psf_d.lower() in ('theo_2d2d', 'theo_uv2d', 'theo_array'):
            psf_Ml = psf.PSF.create_theo_psf(camera, coordinates='st')
            if psf_d.lower() == 'theo_2d2d':
                psf_ml = psf.PSF.create_theo_psf(camera, coordinates='uv')
                self.psf = (psf.PSFApply2D(psf_d=psf_Ml.data, img_size=camera.data_size_ts, use_otf=use_otf), \
                            psf.PSFApply2D(psf_d=psf_ml.data, img_size=self.photo_size_2D, use_otf=use_otf, data_format='raw'))
            elif psf_d.lower() == 'theo_uv2d':
                psf_ml = psf.PSF.create_theo_psf(camera, coordinates='uv')
                self.psf = (psf.PSFApply2D(psf_d=psf_ml.data, img_size=self.photo_size_2D, use_otf=use_otf, data_format='raw'), )
            elif psf_d.lower() == 'theo_array':
                self.psf = (psf.PSFApply2D(psf_d=psf_Ml.data, img_size=camera.data_size_ts, use_otf=use_otf), )
        elif isinstance(psf_d, psf.PSFApply):
            self.psf = (psf_d, )
        else:
            self.psf = psf_d

    def __enter__(self):
        self._initialize_projectors()
        return self

    def __exit__(self, *args):
        self.reset()

    def _initialize_geometry(self, zs):
        """The this function produces the ASTRA geometry needed to refocus or
        project light-fields.

        This function is based on the geometry described in the following articles:
        [1] N. Viganò, et al., “Tomographic approach for the quantitative scene
        reconstruction from light field images,” Opt. Express, vol. 26, no. 18,
        p. 22574, Sep. 2018.
        """
        (samp_v, samp_u, _, _) = self.camera.get_grid_points(space='direct')
        if self.vu_origin is not None:
            (_, _, scale_v, scale_u) = self.camera.get_scales(space='direct', domain=self.domain)
            samp_v -= self.vu_origin[0] * scale_v
            samp_u -= self.vu_origin[1] * scale_u

        (virt_pos_v, virt_pos_u) = np.meshgrid(samp_v, samp_u, indexing='ij')

        if self.shifts_vu[0] is not None:
            virt_pos_v += self.shifts_vu[0]
        if self.shifts_vu[1] is not None:
            virt_pos_u += self.shifts_vu[1]

        if self.domain.lower() == 'object':
            ref_z = self.camera.get_focused_distance()
        elif self.domain.lower() == 'image':
            ref_z = self.camera.z1
        else:
            raise ValueError('No known domain "%s"' % self.domain)

        M = ref_z / self.camera.z1
        camera_pixel_size_ts_us = np.array(self.camera.pixel_size_ts) / self.up_sampling

        renorm_resolution_factor = np.mean(camera_pixel_size_ts_us) * M
        zs_n = zs / renorm_resolution_factor

        voxel_size_obj_space_ts = camera_pixel_size_ts_us * M

        num_imgs = self.camera.get_number_of_subaperture_images()
        imgs_pixel_size_s = np.zeros( (num_imgs, 3) )
        imgs_pixel_size_t = np.zeros( (num_imgs, 3) )
        imgs_pixel_size_s[:, 0] = voxel_size_obj_space_ts[1] / renorm_resolution_factor
        imgs_pixel_size_t[:, 1] = voxel_size_obj_space_ts[0] / renorm_resolution_factor

        if self.camera.is_focused():
            (samp_tau, samp_sigma) = self.camera.get_sigmatau_grid_points(space='direct', domain=self.domain)
            (samp_tau, samp_sigma) = np.meshgrid(samp_tau, samp_sigma, indexing='ij')
            m = - self.camera.a / self.camera.b
            acq_sa_imgs_ts = np.empty((num_imgs, 3))
            acq_sa_imgs_ts[:, 0] = samp_sigma.flatten() * m * M
            acq_sa_imgs_ts[:, 1] = samp_tau.flatten() * m * M
        else:
            acq_sa_imgs_ts = np.zeros((num_imgs, 3))
        acq_sa_imgs_ts[:, 2] = ref_z
        acq_sa_imgs_ts /= renorm_resolution_factor

        ph_imgs_vu = np.empty((num_imgs, 3))
        ph_imgs_vu[:, 0] = virt_pos_u.flatten() / renorm_resolution_factor
        ph_imgs_vu[:, 1] = virt_pos_v.flatten() / renorm_resolution_factor
        ph_imgs_vu[:, 2] = 0

        up_sampled_array_size = np.array(self.camera.data_size_ts) * self.up_sampling
        lims_s = (np.array([-1., 1.]) * up_sampled_array_size[1] / 2) * voxel_size_obj_space_ts[1] / renorm_resolution_factor
        lims_t = (np.array([-1., 1.]) * up_sampled_array_size[0] / 2) * voxel_size_obj_space_ts[0] / renorm_resolution_factor

        num_dists = len(zs_n)
        if self.mode in ('independent', 'simultaneous'):
            self.vol_size = np.concatenate((up_sampled_array_size, (1, )))
        elif self.mode == 'range':
            self.vol_size = np.concatenate((up_sampled_array_size, (num_dists, )))
        else:
            raise ValueError("Mode: '%s' not allowed! Possible choices are: 'independent' | 'simultaneous' | 'range'" % self.mode)

        if self.mode in ('independent', 'simultaneous'):
            self.vol_geom = []
            for d in zs_n:
                lims_z = d + np.array([-1., 1.]) / 2
                self.vol_geom.append(astra.create_vol_geom(self.vol_size[0], self.vol_size[1], self.vol_size[2], lims_s[0], lims_s[1], lims_t[0], lims_t[1], lims_z[0], lims_z[1]))
        elif self.mode == 'range':
            if num_dists > 1:
                lims_z = (zs_n[0], zs_n[-1])
                delta = (lims_z[1] - lims_z[0]) / (num_dists - 1)
                lims_z = lims_z + np.array([-1., 1.]) * delta / 2
            else:
                lims_z = zs_n + np.array([-1., 1.]) / 2
            self.vol_geom = [astra.create_vol_geom(self.vol_size[0], self.vol_size[1], self.vol_size[2], lims_s[0], lims_s[1], lims_t[0], lims_t[1], lims_z[0], lims_z[1])]

        if self.beam_geometry.lower() == 'cone':
            det_geometry = np.hstack([ph_imgs_vu, acq_sa_imgs_ts, imgs_pixel_size_s, imgs_pixel_size_t])
            self.proj_geom = astra.create_proj_geom('cone_vec', self.img_size_us[2], self.img_size_us[3], det_geometry)
        elif self.beam_geometry.lower() == 'parallel':
            proj_dir = acq_sa_imgs_ts - ph_imgs_vu
            img_dist = np.sqrt(np.sum(proj_dir ** 2, axis=1))
            img_dist = np.expand_dims(img_dist, axis=1)
            proj_dir /= img_dist

            det_geometry = np.hstack([proj_dir, acq_sa_imgs_ts, imgs_pixel_size_s, imgs_pixel_size_t])
            self.proj_geom = astra.create_proj_geom('parallel3d_vec', self.img_size_us[2], self.img_size_us[3], det_geometry)
        else:
            raise ValueError("Beam shape: '%s' not allowed! Possible choices are: 'parallel' | 'cone'" % self.beam_geometry)

    def reset(self):
        for p in self.projectors:
            astra.projector.delete(p)
        self.projectors = []

    def _initialize_projectors(self):
        # Volume downscaling option and similar:
        opts = {
            'VoxelSuperSampling': self.super_sampling,
            'DetectorSuperSampling': self.super_sampling,
            'GPUindex': self.gpu_index
            }
        for vg in self.vol_geom:
            proj_id = astra.create_projector('cuda3d', self.proj_geom, vg, opts)
            self.projectors.append(proj_id)
            self.Ws.append(astra.OpTomo(proj_id))

    def FP(self, x):
        """Forward-projection function

        :param x: The volume to project (numpy.array_like)

        :returns: The projected light-field
        :rtype: numpy.array_like
        """
        proj_stack_size = (len(self.Ws), self.img_size_us[-2], self.camera.get_number_of_subaperture_images(), self.img_size_us[-1])

        if self.mode == 'range':
            y = self.Ws[0].FP(np.squeeze(x))
        elif self.mode in ('simultaneous', 'independent'):
            y = np.empty(proj_stack_size, x.dtype)
            for ii, W in enumerate(self.Ws):
                temp = x[ii, :, :]
                temp = W.FP(temp[np.newaxis, ...])
                y[ii, :, :, :] = temp
        y = np.transpose(y, (0, 2, 1, 3))

        for p in self.psf:
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if psf_on_subpixel:
                y = p.apply_psf_direct(y)

        if self.up_sampling > 1:
            y = np.reshape(y, (-1, self.camera.get_number_of_subaperture_images(), self.img_size[-2], self.up_sampling, self.img_size[-1], self.up_sampling))
            y = np.sum(y, axis=(-3, -1)) / (self.up_sampling ** 2)

        y = np.reshape(y, np.concatenate(((-1, ), self.img_size)))

        if self.flat is not None:
            y *= self.flat

        if self.mode == 'simultaneous':
            y = np.sum(y, axis=0)
            y = np.reshape(y, np.concatenate(((-1, ), self.img_size)))

        for p in self.psf:
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if not psf_on_subpixel:
                psf_on_raw = p.data_format is not None and p.data_format.lower() == 'raw'
                if psf_on_raw:
                    # handle 2D flattening
                    y = np.transpose(y, (0, 3, 1, 4, 2))
                    img_size_det = y.shape
                    y = np.reshape(y, np.concatenate(((-1, ), self.photo_size_2D)))

                y = p.apply_psf_direct(y)

                if psf_on_raw:
                    y = np.reshape(y, img_size_det)
                    y = np.transpose(y, (0, 2, 4, 1, 3))

        return y

    def BP(self, y):
        """Back-projection function

        :param x: Light-field to back-project (numpy.array_like)

        :returns: The back-projected volume
        :rtype: numpy.array_like
        """
        y = np.reshape(y, np.concatenate(((-1, ), self.img_size)))

        for p in reversed(self.psf):
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if not psf_on_subpixel:
                psf_on_raw = p.data_format is not None and p.data_format.lower() == 'raw'
                if psf_on_raw:
                    # handle 2D flattening
                    y = np.transpose(y, (0, 3, 1, 4, 2))
                    img_size_det = y.shape
                    y = np.reshape(y, np.concatenate(((-1, ), self.photo_size_2D)))

                y = p.apply_psf_adjoint(y)

                if psf_on_raw:
                    y = np.reshape(y, img_size_det)
                    y = np.transpose(y, (0, 2, 4, 1, 3))

        if self.flat is not None:
            y *= self.flat

        proj_stack_size = np.concatenate(((-1, self.camera.get_number_of_subaperture_images()), self.camera.data_size_ts))
        y = np.reshape(y, proj_stack_size)
        y = np.transpose(y, (0, 2, 1, 3))

        if self.up_sampling > 1:
            y = np.reshape(y, (-1, self.img_size[-2], 1, self.camera.get_number_of_subaperture_images(), self.img_size[-1], 1) )
            y = np.tile(y, [1, 1, self.up_sampling, 1, 1, self.up_sampling])
            y = np.reshape(y, (-1, self.img_size_us[-2], self.camera.get_number_of_subaperture_images(), self.img_size_us[-1]))

        for p in self.psf:
            psf_on_subpixel = p.data_format is not None and p.data_format.lower() == 'subpixel'
            if psf_on_subpixel:
                y = p.apply_psf_direct(y)

        vol_geom_size = (len(self.Ws), self.vol_size[0], self.vol_size[1])

        if self.mode == 'range':
            x = self.Ws[0].BP(np.squeeze(y))
        elif self.mode == 'simultaneous':
            x = np.empty(vol_geom_size, y.dtype)
            for ii, W in enumerate(self.Ws):
                x[ii, :, :] = W.BP(np.squeeze(y))
        else:
            x = np.empty(vol_geom_size, y.dtype)
            for ii, W in enumerate(self.Ws):
                x[ii, :, :] = W.BP(y[ii, :, :, :])

        return x


def compute_forwardprojection(camera : lightfield.Camera, zs, vols, masks, \
                              up_sampling=1, border_padding='edge', \
                              super_sampling=1, border=5, gpu_index=-1, \
                              reflective_geom=True):

    print("Creating projected lightfield..", end='', flush=True)
    c_in = tm.time()

    lf = lightfield.Lightfield(camera, mode='sub-aperture')

    if reflective_geom:
        for ii, z in enumerate(zs):
            dist = np.array((z, ))
            temp_vol = np.expand_dims(vols[ii, :, :], 0)
            temp_mask = np.expand_dims(masks[ii, :, :], 0)
            with Projector(camera, dist, mode='simultaneous', \
                              up_sampling=up_sampling, \
                              super_sampling=super_sampling, \
                              gpu_index=gpu_index) as p:
                pvol = p.FP(temp_vol)
                pmask = p.FP(temp_mask)
                lf.data *= np.squeeze(1 - pmask)
                lf.data += np.squeeze(pvol)
    else:
        with Projector(camera, zs, mode='simultaneous', \
                          up_sampling=up_sampling, \
                          super_sampling=super_sampling, \
                          gpu_index=gpu_index) as p:
            lf.data = p.FP(vols)

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return lf

def compute_refocus_backprojection(lf : lightfield.Lightfield, zs, \
                                   up_sampling=1, border_padding='edge', \
                                   super_sampling=1, border=4, \
                                   beam_geometry='cone', domain='object', \
                                   gpu_index=-1):
    """Compute refocusing of the input lightfield image at the input distances by
    applying the backprojection method.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param super_sampling: Super-sampling of the back-projection operator
        (it will not increase refocused image size/resolution) (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """

    print("Refocusing through Backprojection..", end='', flush=True)
    c_in = tm.time()

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    paddings_st = np.array((border, border))
    lf_sa.pad((0, 0, paddings_st[0], paddings_st[1]), method=border_padding)

    with Projector(lf_sa.camera, zs, flat=lf_sa.flat, mode='range', shifts_vu=lf_sa.shifts_vu, \
                      up_sampling=up_sampling, super_sampling=super_sampling, \
                      gpu_index=gpu_index, beam_geometry=beam_geometry, \
                      domain=domain) as p:
        imgs = p.BP(lf_sa.data)
        ones = p.BP(np.ones_like(lf_sa.data))
        imgs /= ones

    # Crop the refocused images:
    paddings_st = paddings_st * up_sampling
    imgs = imgs[:, paddings_st[0]:-paddings_st[0], paddings_st[1]:-paddings_st[1]]

    c_out = tm.time()
    print("\b\b: Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return imgs

def compute_refocus_iterative(lf : lightfield.Lightfield, zs, iterations=10, \
                              algorithm='sirt', up_sampling=1, \
                              border_padding='edge', super_sampling=1, \
                              gpu_index=-1, beam_geometry='cone', domain='object', \
                              psf=None, vu_origin=None, \
                              use_otf=False, border=4, verbose=False, chunks_zs=1):
    """Compute refocusing of the input lightfield image at the input distances by
    applying iterative methods.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param algorithm: Algorithm to use (string, default: 'sirt')
    :param iterations: Number of iterations (int, default: 10)
    :param psf: Detector PSF (psf.PSFApply, default: None)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param super_sampling: Super-sampling of the back-projection operator
        (it will not increase refocused image size/resolution) (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """

    num_dists = len(zs)
    print("Simultaneous refocusing through %s of %d distances:" % (algorithm.upper(), num_dists))
    c_in = tm.time()

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    paddings_st = np.array((border, border))
    lf_sa.pad((0, 0, paddings_st[0], paddings_st[1]), method=border_padding)

    imgs = np.empty((num_dists, lf_sa.camera.data_size_ts[0] * up_sampling, lf_sa.camera.data_size_ts[1] * up_sampling), dtype=lf_sa.data.dtype)

    c_init = tm.time()

    print(" * Init: %g seconds" % (c_init - c_in))
    for ii_z in range(0, num_dists, chunks_zs):
        print(" * Refocusing chunk of %03d-%03d (avg: %g seconds)" % (ii_z, ii_z+chunks_zs-1, (tm.time() - c_init) / np.fmax(ii_z, 1)))

        sel_zs = zs[ii_z:ii_z+chunks_zs]
        with Projector(lf_sa.camera, sel_zs, flat=lf_sa.flat, psf_d=psf, shifts_vu=lf_sa.shifts_vu, \
                          mode='independent', up_sampling=up_sampling, \
                          super_sampling=super_sampling, use_otf=use_otf, \
                          gpu_index=gpu_index, beam_geometry=beam_geometry, \
                          domain=domain, vu_origin=vu_origin) as p:
            A = lambda x: p.FP(x)
            At = lambda y: p.BP(y)
            b = np.tile(np.reshape(lf_sa.data, np.concatenate(((1, ), p.img_size))), (len(sel_zs), 1, 1, 1, 1))

            if isinstance(algorithm, solvers.Solver):
                algo = algorithm
            elif algorithm.lower() == 'cp_ls':
                algo = solvers.CP_uc(verbose=verbose)
            elif algorithm.lower() == 'cp_tv':
                algo = solvers.CP_tv(verbose=verbose, axes=(-1, -2), lambda_tv=1e-1)
            elif algorithm.lower() == 'cp_wl':
                algo = solvers.CP_wl(verbose=verbose, axes=(-1, -2), lambda_wl=1e-1, wl_type='db1', decomp_lvl=3)
            elif algorithm.lower() == 'bpj':
                algo = solvers.BPJ(verbose=verbose)
            elif algorithm.lower() == 'sirt':
                algo = solvers.Sirt(verbose=verbose)
            else:
                raise ValueError('Unknown algorithm: %s' % algorithm.lower())

            imgs[ii_z:ii_z+len(sel_zs), ...], rel_res_norms = algo(A, b, iterations, At=At, lower_limit=0)

    # Crop the refocused images:
    paddings_st = paddings_st * up_sampling
    imgs = imgs[:, paddings_st[0]:-paddings_st[0], paddings_st[1]:-paddings_st[1]]

    c_out = tm.time()
    print(" * Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return imgs

def compute_refocus_iterative_multiple(lf : lightfield.Lightfield, zs, iterations=10, \
                                       algorithm='sirt', up_sampling=1, \
                                       border_padding='edge', super_sampling=1, \
                                       gpu_index=-1, beam_geometry='cone', domain='object', \
                                       psf=None, vu_origin=None, \
                                       use_otf=True, border=4, verbose=False):
    """Compute refocusing of the input lightfield image simultaneously at the input
    distances by applying iterative methods.

    :param lf: The light-field object (lightfield.Lightfield)
    :param zs: Refocusing distances (numpy.array_like)
    :param algorithm: Algorithm to use (string, default: 'sirt')
    :param iterations: Number of iterations (int, default: 10)
    :param psf: Detector PSF (psf.PSFApply, default: None)
    :param up_sampling: Integer greater than 1 for up-sampling of the final images (int, default: 1)
    :param super_sampling: Super-sampling of the back-projection operator
        (it will not increase refocused image size/resolution) (int, default: 1)
    :param border: Number of pixels to extend the border and reduce darkening of edges (int, default: 4)
    :param border_padding: Border padding method (string, default: 'edge')
    :param beam_geometry: Beam geometry. Possible options: 'parallel' | 'cone' (string, default: 'parallel')
    :param domain: Refocusing domain. Possible options: 'object' | 'image' (string, default: 'object')

    :returns: Stack of 2D refocused images.
    :rtype: numpy.array_like
    """

    print("Simultaneous refocusing through %s:" % algorithm.upper())
    c_in = tm.time()

    lf_sa = lf.clone()
    lf_sa.set_mode_subaperture()

    paddings_st = np.array((border, border))
    lf_sa.pad((0, 0, paddings_st[0], paddings_st[1]), method=border_padding)

    with Projector(lf_sa.camera, zs, flat=lf_sa.flat, psf_d=psf, shifts_vu=lf_sa.shifts_vu, \
                      mode='simultaneous', up_sampling=up_sampling, \
                      super_sampling=super_sampling, use_otf=use_otf, \
                      gpu_index=gpu_index, beam_shape=beam_geometry, \
                      domain=domain, vu_origin=vu_origin) as p:
        A = lambda x: p.FP(x)
        At = lambda y: p.BP(y)
        b = np.reshape(lf_sa.data, np.concatenate(((1, ), p.img_size)))

        if isinstance(algorithm, solvers.Solver):
            algo = algorithm
        elif algorithm.lower() == 'cp_ls':
            algo = solvers.CP_uc(verbose=verbose)
        elif algorithm.lower() == 'cp_tv':
            algo = solvers.CP_tv(verbose=verbose, axes=(-1, -2), lambda_tv=1e-1)
        elif algorithm.lower() == 'cp_wl':
            algo = solvers.CP_wl(verbose=verbose, axes=(-1, -2), lambda_wl=1e-1, wl_type='db1', decomp_lvl=3)
        elif algorithm.lower() == 'bpj':
            algo = solvers.BPJ(verbose=verbose)
        elif algorithm.lower() == 'sirt':
            algo = solvers.Sirt(verbose=verbose)
        else:
            raise ValueError('Unknown algorithm: %s' % algorithm.lower())

        imgs, rel_res_norms = algo(A, b, iterations, At=At, lower_limit=0)

    # Crop the refocused images:
    paddings_st = paddings_st * up_sampling
    imgs = imgs[:, paddings_st[0]:-paddings_st[0], paddings_st[1]:-paddings_st[1]]

    c_out = tm.time()
    print("Done in %g seconds." % (c_out - c_in))

    # Return the stack of refocused images:
    return imgs
