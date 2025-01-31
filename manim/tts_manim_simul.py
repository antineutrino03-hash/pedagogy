from manim import *
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from matplotlib.colors import Normalize
import sys

def multimode_maxwell(
        number_of_modes,
        min_omega,
        max_omega,
        ats,
        ats2,
        ats_max,
        ats2_max,
        min_time=1E-6,
        max_time=1E+4,
        flg=0,
    ):

        gis = np.ones(number_of_modes) * 1E+4                                # Relaxation Moduli
        ti = np.ones(number_of_modes) * 5E+3                                 # Relaxation times
        
        file_path = '/Users/asm18/Documents/python_repo/pyReSpect-freq/output/H.dat'
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
        df.columns = ['Time', 'Relaxation', 'err_lambda', 'err_ols']

        ws = np.logspace(np.log10(min_omega), np.log10(max_omega), 10)
        times = np.logspace(np.log10(min_time), np.log10(max_time), 10)

        # Assume the relaxation frequencies are proportional to omega values
        mk = 6

        gis[4:7] = gis[4:7]
        # gis[0:2] = gis[0:2] * 0.01

        ti1 = ti.copy()
        ti1[:mk] = ti[:mk] * ats
        ti1[mk:] = ti[mk:] * ats2
        
        twi = np.outer(ti1, ws)
        ti_ones = np.ones_like(ti1)
        twi_ones = np.outer(ti_ones, ws)

        ti2 = ti.copy()
        ti2[:mk] = ti[:mk] / ats
        ti2[mk:] = ti[mk:] / ats2
        
        twi2 = np.outer(-1/ti2, times)

        # Prepare arrays for results
        gp_array = np.zeros_like(ws)
        gpp_array = np.zeros_like(ws)
        gt_array = np.zeros_like(ws)

        # Calculate G' and G'' across the omega range for each mode
        for i in range(number_of_modes):
            gp_contribution = gis[i] * (twi[i, :] ** 2 / (1 + twi[i, :] ** 2))
            gpp_contribution = gis[i] * (twi[i, :] / (1 + twi[i, :] ** 2))
            gt_contribution = gis[i] * np.exp(twi2[i, :])

            gp_array += gp_contribution
            gpp_array += gpp_contribution
            gt_array += gt_contribution

        # print(gt_array)
        eta = 00.0
        gt_array += 1E+3
        gpp_array += eta * twi_ones[0, :]
        tan_delta_array = gpp_array / gp_array
        gs_array = np.sqrt(np.square(gpp_array) + np.square(gp_array))

        n = len(ti)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.loglog(
            times,
            gt_array,
            linestyle=" ",
            marker="^",
            markersize=8,
            # label="Reference spectrum: $a_T$",
            color="blue",
        )
        ax.set_xlabel(r"times, $t$", fontsize=18)
        ax.set_ylabel(r"Relaxation moduli, $G_i$", fontsize=18)
        ax.set_title(r"GMM")
        ax.legend(loc="best", fontsize=14)
        plt.tight_layout()
        if flg:
            # plt.savefig(os.path.join('./model_data/', f"spectrum_hd40.png"), dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.close()

        return ws, times, gp_array, gpp_array, gt_array, gs_array, tan_delta_array, n, [gis, ti1]

def wlf_shift_factors(C1, C2, temperatures, T_ref):
        log_shift_factors = -C1 * (temperatures - T_ref) / (C2 + temperatures - T_ref)
        shift_factors = np.power(10, log_shift_factors)
        return shift_factors  # Convert log(a_T) to a_T

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen
        
class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation
        
class SquareAndCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency

        square = Square()  # create a square
        square.set_fill(BLUE, opacity=0.5)  # set the color and transparency

        square.next_to(circle, RIGHT, buff=0.5)  # set the position
        self.play(Create(circle), Create(square))  # show the shapes on screen

class MultiCurvePlot(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 10, 1],
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False                                                              # add log scales here
        )
        axes_labels = axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))
        self.play(Create(axes), Write(axes_labels))

        # Curve 1 Definition
        curve1 = axes.plot(lambda x: 1.0 * x, x_range=[0, 10], color=BLUE)
        trimmed_curve1 = axes.plot(lambda x: 1.0 * x, x_range=[4, 7], color=BLUE)

        # Vertical lines
        temp_point1 = axes.coords_to_point(4.0, 10.0)
        temp_point2 = axes.coords_to_point(7.0, 10.0)

        vline1 = DashedLine(
            start=axes.coords_to_point(4.0, 0),
            end=axes.coords_to_point(4.0, 10),
            dashed_ratio=0.85,
            color=YELLOW
        )
        vline2 = DashedLine(
            start=axes.coords_to_point(7.0, 0),
            end=axes.coords_to_point(7.0, 10),
            dashed_ratio=0.85,
            color=YELLOW
        )

        self.play(Create(curve1))
        self.wait(1)
        self.play(Create(vline1), Create(vline2))
        self.wait(1)

        # Trim to observable range
        obs_axes = Axes(
            x_range=[3, 8, 1], y_range=[3, 8, 1],
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False                                                              # add log scales here
        )
        obs_curve1 = obs_axes.plot(lambda x: 1.0 * x, x_range=[4, 7], color=WHITE)
        obs_axes_labels = obs_axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))

        self.play(
            ReplacementTransform(curve1, trimmed_curve1)
        )
        self.play(FadeOut(vline1), FadeOut(vline2))
        self.play(
            ReplacementTransform(axes, obs_axes),
            ReplacementTransform(axes_labels, obs_axes_labels),
            ReplacementTransform(trimmed_curve1, obs_curve1)
        )
        self.wait(2)

        # add other temperatures
        curves = [
            obs_curve1,
            obs_axes.plot(lambda x: 1.0 * (x + 1), x_range=[4, 7], color=BLUE),
            obs_axes.plot(lambda x: 1.0 * (x - 1), x_range=[4, 7], color=RED)
        ]
        self.play(Create(curves[1]), Create(curves[2]))
        self.wait(2)

        # zoomed out axes
        def transform_point(point):
            coords = obs_axes.p2c(point)
            return axes.c2p(coords[0], coords[1])

        transformed_curves = [curve.copy().apply_function(transform_point) for curve in curves]
        red_axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 10, 1],
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False                                                              # add log scales here
        )
        red_axes_labels = axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))

        self.play(
            ReplacementTransform(obs_axes, red_axes),
            ReplacementTransform(obs_axes_labels, red_axes_labels),
            *[Transform(curve, transformed_curve) for curve, transformed_curve in zip(curves, transformed_curves)]
        )
        self.wait(1)

        reduced_curves = [
            red_axes.plot(lambda x: 1.0 * (x), x_range=[4, 7], color=WHITE),
            red_axes.plot(lambda x: 1.0 * (x), x_range=[5, 8], color=BLUE),
            red_axes.plot(lambda x: 1.0 * (x), x_range=[3, 6], color=RED)
        ]
        r_curve = red_axes.plot(lambda x: 1.0 * (x), x_range=[3, 8], color=WHITE)
        reduced_curve = CurvesAsSubmobjects(r_curve)
        reduced_curve.set_color_by_gradient(RED, BLUE)

        self.play(*[Transform(curve, red_curve) for curve, red_curve in zip(curves, reduced_curves)])
        self.play(*[Transform(curve, reduced_curve) for curve, red_curve in zip(curves, reduced_curves)])
        self.wait(2)
      
class Timer(Scene):
    def construct(self):
        
        temperatures = np.array([273 + 75, 273 + 50, 273 + 25, 273 + 0, 273 - 25, 273 - 50, 273 - 75])
        temperatures = temperatures[::-1]
        T_ref = 273 + 0
        tref_index = np.where(temperatures == T_ref)[0][0]
        C=[5.5, 300]
        number_of_modes = 5
        
        cmap = colormaps['coolwarm'] 
        norm = Normalize(vmin=min(temperatures), vmax=max(temperatures))    

        ws_arr = [[] for i in range((len(temperatures)))]
        ts_arr = [[] for i in range((len(temperatures)))]
        gp_arr = [[] for i in range((len(temperatures)))]
        gpp_arr = [[] for i in range((len(temperatures)))]
        gt_arr = [[] for i in range((len(temperatures)))]
        gs_arr = [[] for i in range((len(temperatures)))]
        tand_arr = [[] for i in range((len(temperatures)))]

        at1 = wlf_shift_factors(C[0], C[1], temperatures, T_ref)
        at2 = wlf_shift_factors(C[0], C[1], temperatures, T_ref)

        for i, at in enumerate(at1):
            ws_arr[i], ts_arr[i], gp_arr[i], gpp_arr[i], gt_arr[i], gs_arr[i], tand_arr[i], n, spectra = multimode_maxwell(
                                                                                                    number_of_modes,
                                                                                                    1E-3,
                                                                                                    1E+6,
                                                                                                    at,
                                                                                                    at2[i],
                                                                                                    np.max(at1),
                                                                                                    np.max(at2),
                                                                                                    flg=0,
                                                                                                    )
        
        ws_arr = np.array(ws_arr)
        ts_arr = np.array(ts_arr)
        gp_arr = np.array(gp_arr)
        gpp_arr = np.array(gpp_arr)
        gt_arr = np.array(gt_arr)
        gs_arr = np.array(gs_arr)
        tand_arr = np.array(tand_arr)
        
        plt.figure(figsize=(8, 6))
        for i, at in enumerate(at1):
            X = ws_arr[i]
            Y1 = gp_arr[i]
            Y2 = gpp_arr[i]

            plt.loglog(
                X,
                Y1,
                label=rf"$G': {temperatures[i]-273}$ $^o$C",
                zorder=1,
            )
            plt.loglog(
                X, Y2, label=rf"$G'': {temperatures[i]-273}$ $^o$C", zorder=2
            )
            
        obs_omega_left = 1E-2
        obs_omega_right = 1E+3
        obs_time_right = 1E+3
        obs_time_left = 1E-2
        moduli_bottom = 1E-6
        moduli_top = 1E+5
        
        # SETTING UP AXES
        freq_axes = Axes(
            x_range=[np.log10(np.min(ws_arr))-1, np.log10(np.max(ws_arr))+1, 2],     # Logarithmic range for the x-axis
            y_range=[np.log10(np.min(gpp_arr))-1, np.log10(np.max(gpp_arr))+1, 2],     # Logarithmic range for the y-axis
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(custom_labels=True)},
            y_axis_config={"scaling": LogBase(custom_labels=True)}
        ).to_edge(LEFT, buff=1)
        
        time_axes = Axes(
            x_range=[np.log10(np.min(ts_arr))-1, np.log10(np.max(ts_arr))+1, 2],     # Logarithmic range for the x-axis
            y_range=[np.log10(np.min(gt_arr))-1, np.log10(np.max(gt_arr))+1, 2],     # Logarithmic range for the y-axis
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(custom_labels=True)},
            y_axis_config={"scaling": LogBase(custom_labels=True)}
        ).to_edge(RIGHT, buff=1)
        
        # AXES LABELS
        freq_axes_labels = freq_axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))
        self.play(Create(freq_axes), Write(freq_axes_labels))
        time_axes_labels = time_axes.get_axis_labels(x_label=MathTex(r"t"), y_label=MathTex(r"G(t)"))
        self.play(Create(time_axes), Write(time_axes_labels))

        # GROUND TRUTH
        master_freq_dots = [
            Dot(point=freq_axes.c2p(ws_arr[tref_index][j], gpp_arr[tref_index][j]), color=BLUE)
            for j in np.arange(len(ws_arr[tref_index]))
        ]
        master_time_dots = [
            Dot(point=time_axes.c2p(ts_arr[tref_index][j], gt_arr[tref_index][j]), color=BLUE)
            for j in np.arange(len(ws_arr[tref_index]))
        ]

        # TRIM TO OBSERVABLE FREQ
        freq_vline1 = DashedLine(
            start=freq_axes.coords_to_point(obs_omega_left, moduli_bottom),
            end=freq_axes.coords_to_point(obs_omega_left, moduli_top),
            dashed_ratio=0.85,
            color=YELLOW
        )
        freq_vline2 = DashedLine(
            start=freq_axes.coords_to_point(obs_omega_right, moduli_bottom),
            end=freq_axes.coords_to_point(obs_omega_right, moduli_top),
            dashed_ratio=0.85,
            color=YELLOW
        )
        master_trimmed_freq = [
            Dot(point=freq_axes.c2p(ws_arr[tref_index][j], gpp_arr[tref_index][j]), color=BLUE)
            for j in np.arange(len(ws_arr[tref_index]))
            if obs_omega_left <= ws_arr[tref_index][j] <= obs_omega_right
        ]
        
        # TRIM TO OBSERVABLE TIME
        time_vline1 = DashedLine(
            start=freq_axes.coords_to_point(obs_time_right, moduli_bottom),
            end=freq_axes.coords_to_point(obs_time_right, moduli_top),
            dashed_ratio=0.85,
            color=YELLOW
        )
        time_vline2 = DashedLine(
            start=time_axes.coords_to_point(obs_time_left, moduli_bottom),
            end=time_axes.coords_to_point(obs_time_left, moduli_top),
            dashed_ratio=0.85,
            color=YELLOW
        )
        master_trimmed_time = [
            Dot(point=time_axes.c2p(ts_arr[tref_index][j], gt_arr[tref_index][j]), color=BLUE)
            for j in np.arange(len(ws_arr[tref_index]))
            if obs_time_left <= ts_arr[tref_index][j] <= obs_time_right
        ]

        # ANIMATIONS
        self.play(*[Create(dot) for dot in master_freq_dots])
        self.play(*[Create(dot) for dot in master_time_dots])
        self.wait(1) 
        self.play(Create(freq_vline1), Create(freq_vline2))
        self.play(Create(time_vline1), Create(time_vline2))
        
        # OBSERVABLE RANGE AXES
        obs_freq_axes = Axes(
            x_range=[np.log10(obs_omega_left)-1, np.log10(obs_omega_right)+1, 1],  
            y_range=[np.log10(np.min(gpp_arr))-1, np.log10(np.max(gpp_arr))+1, 2],   
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(10, custom_labels=True)},
            y_axis_config={"scaling": LogBase(10, custom_labels=True)}
        ).to_edge(LEFT, buff=1)
        obs_freq_labels = obs_freq_axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))
        
        obs_time_axes = Axes(
            x_range=[np.log10(obs_time_left)-1, np.log10(obs_time_right)+1, 1],  
            y_range=[np.log10(np.min(gt_arr))-1, np.log10(np.max(gt_arr))+1, 2],   
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(10, custom_labels=True)},
            y_axis_config={"scaling": LogBase(10, custom_labels=True)}
        ).to_edge(RIGHT, buff=1)
        obs_time_labels = obs_time_axes.get_axis_labels(x_label=MathTex(r"t"), y_label=MathTex(r"G(t)"))
        
        color = cmap(norm(T_ref))
        color = rgb_to_color(color[:3])
        
        obs_freq_ref = [
            Dot(point=obs_freq_axes.c2p(ws_arr[tref_index][j], gpp_arr[tref_index][j]), color=color)
            for j in np.arange(len(ws_arr[tref_index]))
            if obs_omega_left <= ws_arr[tref_index][j] <= obs_omega_right
        ]
        obs_time_ref = [
            Dot(point=obs_time_axes.c2p(ts_arr[tref_index][j], gt_arr[tref_index][j]), color=color)
            for j in np.arange(len(ws_arr[tref_index]))
            if obs_time_left <= ts_arr[tref_index][j] <= obs_time_right
        ]
        
        self.play(
            *[FadeOut(dot) for dot in master_freq_dots],
            *[FadeOut(dot) for dot in master_time_dots],
            *[FadeIn(dot) for dot in master_trimmed_freq],
            *[FadeIn(dot) for dot in master_trimmed_time],
        )
        
        self.play(FadeOut(freq_vline1), FadeOut(freq_vline2))
        self.play(FadeOut(time_vline1), FadeOut(time_vline2))
        
        self.play(
            ReplacementTransform(freq_axes, obs_freq_axes),
            ReplacementTransform(time_axes, obs_time_axes),
            ReplacementTransform(freq_axes_labels, obs_freq_labels),
            ReplacementTransform(time_axes_labels, obs_time_labels),
            *[ReplacementTransform(trimmed_dot, obs_dot) for trimmed_dot, obs_dot in zip(master_trimmed_freq, obs_freq_ref)],
            *[ReplacementTransform(trimmed_dot, obs_dot) for trimmed_dot, obs_dot in zip(master_trimmed_time, obs_time_ref)],
        )
        self.wait(2)
        
        # ADD OTHER TEMPERATURES

        obs_freq_curves = []
        obs_time_curves = []
        
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                obs_freq_curve = [
                    Dot(point=obs_freq_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
                obs_time_curve = [
                    Dot(point=obs_time_axes.c2p(ts_arr[i][j], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            if temp < T_ref:
                obs_freq_curve = [
                    Dot(point=obs_freq_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
                obs_time_curve = [
                    Dot(point=obs_time_axes.c2p(ts_arr[i][j], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            if temp == T_ref:
                obs_freq_curve = obs_freq_ref
                obs_time_curve = obs_time_ref
                
            obs_freq_curves.append(obs_freq_curve)
            obs_time_curves.append(obs_time_curve)
        
        for obs_freq_curve in obs_freq_curves:
            if obs_freq_curve is not obs_freq_ref:    
                self.play(*[
                    Create(dot)
                    for dot in obs_freq_curve
                ])
                self.wait(0.5)
                
        for obs_time_curve in obs_time_curves:
            if obs_time_curve is not obs_freq_ref:    
                self.play(*[
                    Create(dot)
                    for dot in obs_time_curve
                ])
                self.wait(0.5)
        
        def transform_point(point):
            coords = obs_axes.p2c(point)
            return axes.c2p(coords[0], coords[1])
        
        # REDUCED MASTERCURVE AXES
        
        red_freq_axes = Axes(
            x_range=[np.log10(obs_omega_left * np.min(at1))-1, np.log10(obs_omega_right * np.max(at1))+1, 2], 
            y_range=[np.log10(np.min(gpp_arr))-1, np.log10(np.max(gpp_arr))+1, 2],    
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(10, custom_labels=True)},
            y_axis_config={"scaling": LogBase(10, custom_labels=True)}
        ).to_edge(LEFT, buff=1)
        red_freq_labels = red_freq_axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))
        
        red_time_axes = Axes(
            x_range=[np.log10(obs_time_left / np.max(at1))-1, np.log10(obs_time_right / np.min(at1))+1, 2], 
            y_range=[np.log10(np.min(gt_arr))-1, np.log10(np.max(gt_arr))+1, 2],    
            x_length=5, y_length=4,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(10, custom_labels=True)},
            y_axis_config={"scaling": LogBase(10, custom_labels=True)}
        ).to_edge(RIGHT, buff=1)
        red_time_labels = red_time_axes.get_axis_labels(x_label=MathTex(r"t"), y_label=MathTex(r"G(t)"))
        
        # REDUCED MASTER CURVES
        
        reduced_freq_curves = []
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                reduced_freq_curve = [
                    Dot(point=red_freq_axes.c2p(ws_arr[i][j]*at1[i], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp < T_ref:
                reduced_freq_curve = [
                    Dot(point=red_freq_axes.c2p(ws_arr[i][j]*at1[i], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp == T_ref:
                reduced_freq_curve = [
                    Dot(point=red_freq_axes.c2p(ws_arr[i][j]*at1[i], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            reduced_freq_curves.append(reduced_freq_curve)
            
        reduced_time_curves = []
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                reduced_time_curve = [
                    Dot(point=red_time_axes.c2p(ts_arr[i][j]/at1[i], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ts_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            if temp < T_ref:
                reduced_time_curve = [
                    Dot(point=red_time_axes.c2p(ts_arr[i][j]/at1[i], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ts_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            if temp == T_ref:
                reduced_time_curve = [
                    Dot(point=red_time_axes.c2p(ts_arr[i][j]/at1[i], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ts_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            reduced_time_curves.append(reduced_time_curve)
            
        # ZOOMED OUT CURVES    
            
        outzoomed_freq_curves = []
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                outzoomed_freq_curve = [
                    Dot(point=red_freq_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp < T_ref:
                outzoomed_freq_curve = [
                    Dot(point=red_freq_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp == T_ref:
                outzoomed_freq_curve = [
                    Dot(point=red_freq_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            outzoomed_freq_curves.append(outzoomed_freq_curve)
            
        outzoomed_time_curves = []
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                outzoomed_time_curve = [
                    Dot(point=red_time_axes.c2p(ts_arr[i][j], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ts_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            if temp < T_ref:
                outzoomed_time_curve = [
                    Dot(point=red_time_axes.c2p(ts_arr[i][j], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ts_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            if temp == T_ref:
                outzoomed_time_curve = [
                    Dot(point=red_time_axes.c2p(ts_arr[i][j], gt_arr[i][j]), color=color)
                    for j in np.arange(len(ts_arr[i]))
                    if obs_time_left <= ts_arr[i][j] <= obs_time_right
                ]
            outzoomed_time_curves.append(outzoomed_time_curve)
            
        self.play(
            ReplacementTransform(obs_freq_axes, red_freq_axes),
            ReplacementTransform(obs_time_axes, red_time_axes),
            ReplacementTransform(obs_freq_labels, red_freq_labels),
            ReplacementTransform(obs_time_labels, red_time_labels),
            *[
                Transform(obs_dot, transformed_dot)
                for obs_curve, transformed_curve in zip(obs_freq_curves, outzoomed_freq_curves)
                for obs_dot, transformed_dot in zip(obs_curve, transformed_curve)
            ],
            *[
                Transform(obs_dot, transformed_dot)
                for obs_curve, transformed_curve in zip(obs_time_curves, outzoomed_time_curves)
                for obs_dot, transformed_dot in zip(obs_curve, transformed_curve)
            ]
        )
        
        for obs_freq_curve, transformed_curve in zip(obs_freq_curves, reduced_freq_curves):
            if obs_freq_curve is not obs_freq_ref: 
                self.play(*[Transform(obs_dot, transformed_dot)
                            for obs_dot, transformed_dot in zip(obs_freq_curve, transformed_curve)
                            ])
                self.wait(0.5)
                
        for obs_time_curve, transformed_curve in zip(obs_time_curves, reduced_time_curves):
            if obs_time_curve is not obs_time_ref: 
                self.play(*[Transform(obs_dot, transformed_dot)
                            for obs_dot, transformed_dot in zip(obs_time_curve, transformed_curve)
                            ])
                self.wait(0.5)
        self.wait(2)