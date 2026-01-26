''' Snap4city Computing HEATMAP - Data Interpolation Module.
    Copyright (C) 2026 DISIT Lab http://www.disit.org - University of Florence
'''

import math
import numpy as np
import logging
from abc import ABC, abstractmethod
from shapely.geometry import MultiPoint
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Akima1DInterpolator
from shapely import contains_xy
import logic.config as config

# Standard logging configuration
logger = logging.getLogger(__name__)

class Interpolator(ABC):
    """
    Abstract Base Class for spatial interpolation strategies.
    
    This class provides a framework for generating a coordinate grid from 
    geographic boundaries and performing post-interpolation refinement. 
    It ensures that computational complexity stays within defined limits.
    """
    def __init__(self, xy_known, val_known, max_cells=None, base_cell_size=None, bbox=None):
        """
        Initializes the Interpolator with sensor data and spatial constraints.

        Args:
            xy_known (np.ndarray): Nx2 array of sensor coordinates in UTM (meters).
            val_known (np.ndarray): N-length array of observed sensor values.
            max_cells (int): Maximum number of active cells allowed for computation.
            base_cell_size (float): Initial target resolution (step size) in meters.
            bbox (shapely.geometry.Polygon): Polygon defining the total output area.
        """
        self.xy_known = xy_known
        self.val_known = val_known
        self.max_cells = max_cells if max_cells is not None else config.MAX_CELLS
        self.cell_size = base_cell_size if base_cell_size is not None else config.BASE_CELL_SIZE
        self.bbox = bbox
        self.sigma = config.GAUSSIAN_SIGMA

    @staticmethod
    def build(method, xy_known, val_known, max_cells, bbox):
        """
        Factory method to instantiate the appropriate Interpolator subclass.

        Args:
            method (str): The interpolation algorithm to use ('idw' or 'akima').
            xy_known (np.ndarray): Known point coordinates.
            val_known (np.ndarray): Values at known points.
            max_cells (int): Cell limit for the active calculation area.
            bbox (shapely.geometry.Polygon): Bounding Box for the output.

        Returns:
            Interpolator: An instance of the requested Interpolator subclass.
        
        Raises:
            ValueError: If an unsupported interpolation method is specified.
        """
        logger.info(f"Building interpolator instance for method: {method}")
        if method == 'idw':
            return IDWInterpolator(xy_known, val_known, max_cells, bbox=bbox)
        elif method == 'akima':
            return AkimaInterpolator(xy_known, val_known, max_cells, bbox=bbox)
        else:
            logger.error(f"Method {method} is not supported")
            raise ValueError(f"Unknown method: {method}")

    def compute_step_from_area(self, area_meters, target_limit=None):
        """
        Calculates the theoretical step size required to fit an area into a cell limit.

        Args:
            area_meters (float): The surface area to cover in square meters.
            target_limit (int, optional): The maximum number of cells. Defaults to self.max_cells.

        Returns:
            float: The calculated ideal step size in meters.
        """
        limit = target_limit if target_limit else self.max_cells
        ideal_step = math.sqrt(area_meters / limit)
        final_step = max(self.cell_size, ideal_step)
        
        logger.info(f"Step optimization: Area={area_meters:.2f}m2, Limit={limit} -> Ideal Step={final_step:.2f}m")
        return math.ceil(final_step)

    def run(self):
        """
        Executes the full interpolation workflow.

        Returns:
            tuple: A 4-element tuple containing:
                - grid_x (np.ndarray): 2D array of X coordinates.
                - grid_y (np.ndarray): 2D array of Y coordinates.
                - grid_z (np.ndarray): 2D array of smoothed interpolated values.
                - cell_size (float): The final step size used for the grid.
        """
        logger.info("Starting interpolation workflow")
        
        grid_x, grid_y = self.build_grid()
        logger.info(f"Step 1: Grid generated. Shape: {grid_x.shape}, Step: {self.cell_size}m")
        
        grid_z_raw = self.interpolate(grid_x, grid_y)
        logger.info("Step 2: Core interpolation completed")
        
        grid_z_final = self.post_process(grid_x, grid_y, grid_z_raw)
        logger.info("Step 3: Post-processing (Gaussian smoothing) completed")
        
        return grid_x, grid_y, grid_z_final, self.cell_size

    def post_process(self, grid_x, grid_y, grid_z):
        """
        Applies a Gaussian filter to the interpolated grid to smooth transitions.

        Args:
            grid_x (np.ndarray): Grid X coordinates.
            grid_y (np.ndarray): Grid Y coordinates.
            grid_z (np.ndarray): Raw interpolated Z values (may contain NaNs).

        Returns:
            np.ndarray: Smoothed Z values with the original NaN mask restored.
        """
        logger.info(f"Post-processing: Applying Gaussian filter (sigma={self.sigma})")
        nan_mask = np.isnan(grid_z)
        if not np.any(~nan_mask):
            logger.warning("Post-processing: Input grid is empty (all NaNs)")
            return grid_z

        # Fill NaNs with the mean value to prevent edge bleeding during blur
        grid_z_filled = np.where(nan_mask, np.nanmean(grid_z), grid_z)
        smoothed_z = gaussian_filter(grid_z_filled, sigma=self.sigma)
        
        # Restore original NaNs for areas outside the interpolation mask
        smoothed_z[nan_mask] = np.nan
        return smoothed_z

    @abstractmethod
    def build_grid(self):
        """Generates the meshgrid based on the Bounding Box and optimized step."""
        pass

    @abstractmethod
    def interpolate(self, grid_x, grid_y):
        """Performs the specific mathematical interpolation logic."""
        pass

class IDWInterpolator(Interpolator):
    """
    Inverse Distance Weighting (IDW) Interpolator.
    
    Generates a continuous surface across the entire Bounding Box, with 
    values fading to zero as distance from the nearest sensor increases.
    """
    def build_grid(self):
        """
        Calculates a uniform grid over the Bounding Box.
        
        Returns:
            tuple: (grid_x, grid_y) as generated by np.meshgrid.
        """
        logger.info("IDW: Calculating step for BBox area")
        xmin, ymin, xmax, ymax = self.bbox.bounds
        bbox_area = (xmax - xmin) * (ymax - ymin)
        self.cell_size = self.compute_step_from_area(bbox_area)
        
        x_edges = np.arange(xmin, xmax + self.cell_size, self.cell_size)
        y_edges = np.arange(ymin, ymax + self.cell_size, self.cell_size)
        return np.meshgrid(x_edges, y_edges)

    def interpolate(self, grid_x, grid_y):
        """
        Calculates IDW values with configurable power and decay threshold.

        Args:
            grid_x (np.ndarray): Meshgrid X.
            grid_y (np.ndarray): Meshgrid Y.

        Returns:
            np.ndarray: 2D array of interpolated values.
        """
        logger.info(f"IDW: Processing matrix of shape {grid_x.shape}")
        gx, gy = grid_x.flatten(), grid_y.flatten()
        pts = np.vstack((gx, gy)).T
        
        # Broadcasting distance calculation
        dist = np.sqrt(((pts[:, None, :] - self.xy_known[None, :, :]) ** 2).sum(axis=2))
        min_dist = np.min(dist, axis=1)
        
        dist[dist == 0] = 1e-10  # Prevent division by zero
        w = 1 / dist**config.IDW_POWER
        z = np.sum(w * self.val_known[None, :], axis=1) / w.sum(axis=1)
        
        # Exponential fade-out beyond configured meters
        fade = np.where(
            min_dist <= config.IDW_FADE_DISTANCE, 
            1.0, 
            np.exp(-(min_dist - config.IDW_FADE_DISTANCE) / (config.IDW_FADE_SMOOTHING * self.cell_size))
        )
        return (z * fade).reshape(grid_x.shape)

class AkimaInterpolator(Interpolator):
    """
    2D Akima Spline Interpolator.
    
    Uses a dual-pass (row-wise and column-wise) 1D Akima spline approach. 
    Resolution is prioritized within the convex hull of sensors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-calculate the convex hull of the sensor network
        self.hull_orig = MultiPoint(self.xy_known).convex_hull

    def build_grid(self):
        """
        Optimizes resolution specifically for the area inside the convex hull.
        Increments cell size iteratively if the active cell count exceeds max_cells.

        Returns:
            tuple: (grid_x, grid_y) meshgrid.
        """
        logger.info("Akima: Starting iterative resolution optimization")
        xmin, ymin, xmax, ymax = self.bbox.bounds
        
        # Initial step calculation based on hull area
        self.cell_size = self.compute_step_from_area(self.hull_orig.area)
        
        while True:
            # Generate temporary grid to count active cells
            x_edges = np.arange(xmin, xmax + self.cell_size, self.cell_size)
            y_edges = np.arange(ymin, ymax + self.cell_size, self.cell_size)
            gx, gy = np.meshgrid(x_edges, y_edges)
            
            # Use a buffer to include sensors near the hull edge
            hull_buffer = self.hull_orig.buffer(self.cell_size * config.AKIMA_HULL_BUFFER_FACTOR)
            active_cells = np.sum(contains_xy(hull_buffer, gx, gy))
            
            if active_cells <= self.max_cells:
                logger.info(f"Akima: Limit respected. Final Step={self.cell_size}m, Active Cells={active_cells}")
                break
            
            # Incrementally increase step to reduce cell count
            self.cell_size += 1
            logger.info(f"Akima: Limit exceeded ({active_cells} > {self.max_cells}). Retrying with step {self.cell_size}m")
        
        return np.meshgrid(x_edges, y_edges)

    def interpolate(self, grid_x, grid_y):
        """
        Performs 2D interpolation using row-column passes of Akima splines.

        Args:
            grid_x (np.ndarray): Meshgrid X.
            grid_y (np.ndarray): Meshgrid Y.

        Returns:
            np.ndarray: 2D array of interpolated values within the hull.
        """
        logger.info("Akima: Performing dual-pass spline calculation")
        hull_buffer = self.hull_orig.buffer(self.cell_size * config.AKIMA_HULL_BUFFER_FACTOR)
        mask = contains_xy(hull_buffer, grid_x, grid_y)
        
        tree = cKDTree(self.xy_known)
        z_rows, z_cols = np.full(grid_x.shape, np.nan), np.full(grid_x.shape, np.nan)
        rows, cols = np.where(mask)
        
        if rows.size == 0:
            logger.warning("Akima: No active cells within hull mask")
            return z_rows

        # Row-wise interpolation pass
        for r in range(rows.min(), rows.max() + 1):
            idx = np.where(mask[r, :])[0]
            if len(idx) < 2: continue
            x_c = np.linspace(grid_x[r, idx[0]], grid_x[r, idx[-1]], config.AKIMA_SPLINE_POINTS)
            d, i = tree.query(np.column_stack((x_c, np.full(config.AKIMA_SPLINE_POINTS, grid_y[r, 0]))), k=min(config.AKIMA_K_NEIGHBORS, len(self.xy_known)))
            w = 1.0 / ((d + 1e-12)**config.AKIMA_WEIGHT_POWER)
            z_c = np.sum(w * self.val_known[i], axis=1) / np.sum(w, axis=1)
            try: z_rows[r, idx] = Akima1DInterpolator(x_c, z_c)(grid_x[r, idx])
            except Exception: z_rows[r, idx] = np.interp(grid_x[r, idx], x_c, z_c)

        # Column-wise interpolation pass
        for c in range(cols.min(), cols.max() + 1):
            idx = np.where(mask[:, c])[0]
            if len(idx) < 2: continue
            y_c = np.linspace(grid_y[idx[0], c], grid_y[idx[-1], c], config.AKIMA_SPLINE_POINTS)
            d, i = tree.query(np.column_stack((np.full(config.AKIMA_SPLINE_POINTS, grid_x[0, c]), y_c)), k=min(config.AKIMA_K_NEIGHBORS, len(self.xy_known)))
            w = 1.0 / ((d + 1e-12)**config.AKIMA_WEIGHT_POWER)
            z_c = np.sum(w * self.val_known[i], axis=1) / np.sum(w, axis=1)
            try: z_cols[idx, c] = Akima1DInterpolator(y_c, z_c)(grid_y[idx, c])
            except Exception: z_cols[idx, c] = np.interp(grid_y[idx, c], y_c, z_c)

        # Combine passes using nanmean to handle potential edge artifacts
        with np.errstate(all='ignore'): 
            grid_z = np.nanmean([z_rows, z_cols], axis=0)
        
        grid_z[~mask] = np.nan
        logger.info("Akima: Core interpolation pass finished")
        return grid_z