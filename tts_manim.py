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
        flg=0,
    ):

        gis = np.ones(number_of_modes) * 1E+4                                # Relaxation Moduli
        ti = np.ones(number_of_modes) * 5E-2                                 # Relaxation times
        
        file_path = '/Users/asm18/Documents/python_repo/pyReSpect-freq/output/H.dat'
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
        df.columns = ['Time', 'Relaxation', 'err_lambda', 'err_ols']

        ws = np.logspace(np.log10(min_omega), np.log10(max_omega), 30)

        # Assume the relaxation frequencies are proportional to omega values
        mk = 6
        ti1 = ti.copy()
        ti1[:mk] = ti[:mk] * ats
        ti1[mk:] = ti[mk:] * ats2

        gis[4:7] = gis[4:7]
        # gis[0:2] = gis[0:2] * 0.01

        twi = np.outer(ti1, ws)
        ti_ones = np.ones_like(ti1)
        twi_ones = np.outer(ti_ones, ws)

        ti2 = ti.copy()
        ti2[:mk] = ti[:mk] * ats_max
        ti2[mk:] = ti[mk:] * ats2_max
        # twi = np.outer(ti2, ws)

        # Prepare arrays for results
        gp_array = np.zeros_like(ws)
        gpp_array = np.zeros_like(ws)

        # Calculate G' and G'' across the omega range for each mode
        for i in range(number_of_modes):
            gp_contribution = gis[i] * (twi[i, :] ** 2 / (1 + twi[i, :] ** 2))
            gpp_contribution = gis[i] * (twi[i, :] / (1 + twi[i, :] ** 2))

            gp_array += gp_contribution
            gpp_array += gpp_contribution

        eta = 00.0
        gpp_array += eta * twi_ones[0, :]
        tan_delta_array = gpp_array / gp_array
        gs_array = np.sqrt(np.square(gpp_array) + np.square(gp_array))

        n = len(ti)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.loglog(
            ti1,
            gis,
            linestyle=" ",
            marker="^",
            markersize=8,
            # label="Reference spectrum: $a_T$",
            color="blue",
        )
        ax.set_xlabel(r"Relaxation times, $\tau_i$", fontsize=18)
        ax.set_ylabel(r"Relaxation moduli, $G_i$", fontsize=18)
        ax.set_title(r"GMM")
        ax.legend(loc="best", fontsize=14)
        plt.tight_layout()
        if flg:
            # plt.savefig(os.path.join('./model_data/', f"spectrum_hd40.png"), dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.close()

        return ws, gp_array, gpp_array, gs_array, tan_delta_array, n, [gis, ti1]

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
            x_length=7, y_length=5,
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
            x_length=7, y_length=5,
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
            x_length=7, y_length=5,
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
        gp_arr = [[] for i in range((len(temperatures)))]
        gpp_arr = [[] for i in range((len(temperatures)))]
        gs_arr = [[] for i in range((len(temperatures)))]
        tand_arr = [[] for i in range((len(temperatures)))]

        at1 = wlf_shift_factors(C[0], C[1], temperatures, T_ref)
        at2 = wlf_shift_factors(C[0], C[1], temperatures, T_ref)

        for i, at in enumerate(at1):
            ws_arr[i], gp_arr[i], gpp_arr[i], gs_arr[i], tand_arr[i], n, spectra = multimode_maxwell(
                                                                                            number_of_modes,
                                                                                            1E-6,
                                                                                            1E+6,
                                                                                            at,
                                                                                            at2[i],
                                                                                            np.max(at1),
                                                                                            np.max(at2),
                                                                                            flg=0,
                                                                                            )
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
        moduli_bottom = 1E-6
        moduli_top = 1E+7
        
        axes = Axes(
            x_range=[np.log10(np.min(ws_arr))-1, np.log10(np.max(ws_arr))+1, 2],     # Logarithmic range for the x-axis
            y_range=[np.log10(np.min(gpp_arr))-1, np.log10(np.max(gpp_arr))+1, 2],     # Logarithmic range for the y-axis
            x_length=7, y_length=5,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(custom_labels=True)},
            y_axis_config={"scaling": LogBase(custom_labels=True)}
        )
        axes_labels = axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))
        self.play(Create(axes), Write(axes_labels))

        dots = [
            Dot(point=axes.c2p(ws_arr[tref_index][j], gpp_arr[tref_index][j]), color=BLUE)
            for j in np.arange(len(ws_arr[tref_index]))
        ]

        vline1 = DashedLine(
            start=axes.coords_to_point(obs_omega_left, moduli_bottom),
            end=axes.coords_to_point(obs_omega_left, moduli_top),
            dashed_ratio=0.85,
            color=YELLOW
        )
        vline2 = DashedLine(
            start=axes.coords_to_point(obs_omega_right, moduli_bottom),
            end=axes.coords_to_point(obs_omega_right, moduli_top),
            dashed_ratio=0.85,
            color=YELLOW
        )
        trimmed_dots = [
            Dot(point=axes.c2p(ws_arr[tref_index][j], gpp_arr[tref_index][j]), color=BLUE)
            for j in np.arange(len(ws_arr[tref_index]))
            if obs_omega_left <= ws_arr[tref_index][j] <= obs_omega_right
        ]

        self.play(*[Create(dot) for dot in dots])
        self.wait(1) 
        self.play(Create(vline1), Create(vline2))
        
        obs_axes = Axes(
            x_range=[np.log10(obs_omega_left)-1, np.log10(obs_omega_right)+1, 1],     # Logarithmic range for the x-axis
            y_range=[np.log10(np.min(gpp_arr))-1, np.log10(np.max(gpp_arr))+1, 2],     # Logarithmic range for the y-axis
            x_length=7, y_length=5,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(10, custom_labels=True)},
            y_axis_config={"scaling": LogBase(10, custom_labels=True)}
        )
        
        color = cmap(norm(T_ref))
        color = rgb_to_color(color[:3])
        obs_curve1 = [
            Dot(point=obs_axes.c2p(ws_arr[tref_index][j], gpp_arr[tref_index][j]), color=color)
            for j in np.arange(len(ws_arr[tref_index]))
            if obs_omega_left <= ws_arr[tref_index][j] <= obs_omega_right
        ]
        obs_axes_labels = obs_axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))
        
        self.play(
            *[FadeOut(dot) for dot in dots],
            *[FadeIn(dot) for dot in trimmed_dots]
        )
        self.play(FadeOut(vline1), FadeOut(vline2))
        self.play(
            ReplacementTransform(axes, obs_axes),
            ReplacementTransform(axes_labels, obs_axes_labels),
            *[ReplacementTransform(trimmed_dot, obs_dot) for trimmed_dot, obs_dot in zip(trimmed_dots, obs_curve1)]
        )
        self.wait(2)
        
        # add other temperatures
        obs_curves = []
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                obs_curve = [
                    Dot(point=obs_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp < T_ref:
                obs_curve = [
                    Dot(point=obs_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp == T_ref:
                obs_curve = obs_curve1
            obs_curves.append(obs_curve)
        
        for obs_curve in obs_curves:
            if obs_curve is not obs_curve1:    
                self.play(*[
                    Create(dot)
                    for dot in obs_curve
                ])
                self.wait(0.5)
        
        def transform_point(point):
            coords = obs_axes.p2c(point)
            return axes.c2p(coords[0], coords[1])
        
        red_axes = Axes(
            x_range=[np.log10(obs_omega_left * np.min(at1))-1, np.log10(obs_omega_right * np.max(at1))+1, 2],     # Logarithmic range for the x-axis
            y_range=[np.log10(np.min(gpp_arr))-1, np.log10(np.max(gpp_arr))+1, 2],     # Logarithmic range for the y-axis
            x_length=7, y_length=5,
            axis_config={"include_numbers": True},
            tips=False,
            x_axis_config={"scaling": LogBase(10, custom_labels=True)},
            y_axis_config={"scaling": LogBase(10, custom_labels=True)}
        )
        red_axes_labels = axes.get_axis_labels(x_label=MathTex(r"\omega"), y_label=MathTex(r"G''(\omega)"))
        
        reduced_curves = []
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                reduced_curve = [
                    Dot(point=red_axes.c2p(ws_arr[i][j]*at1[i], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp < T_ref:
                reduced_curve = [
                    Dot(point=red_axes.c2p(ws_arr[i][j]*at1[i], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp == T_ref:
                reduced_curve = [
                    Dot(point=red_axes.c2p(ws_arr[i][j]*at1[i], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            reduced_curves.append(reduced_curve)
            
        outzoomed_curves = []
        for i, temp in enumerate(temperatures):
            color = cmap(norm(temp))
            color = rgb_to_color(color[:3])
            if temp > T_ref:
                outzoomed_curve = [
                    Dot(point=red_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp < T_ref:
                outzoomed_curve = [
                    Dot(point=red_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            if temp == T_ref:
                outzoomed_curve = [
                    Dot(point=red_axes.c2p(ws_arr[i][j], gpp_arr[i][j]), color=color)
                    for j in np.arange(len(ws_arr[i]))
                    if obs_omega_left <= ws_arr[i][j] <= obs_omega_right
                ]
            outzoomed_curves.append(outzoomed_curve)
            
        self.play(
            ReplacementTransform(obs_axes, red_axes),
            ReplacementTransform(obs_axes_labels, red_axes_labels),
            *[
                Transform(obs_dot, transformed_dot)
                for obs_curve, transformed_curve in zip(obs_curves, outzoomed_curves)
                for obs_dot, transformed_dot in zip(obs_curve, transformed_curve)
            ]
        )
        for obs_curve, transformed_curve in zip(obs_curves, reduced_curves):
            if obs_curve is not obs_curve1: 
                self.play(*[Transform(obs_dot, transformed_dot)
                            for obs_dot, transformed_dot in zip(obs_curve, transformed_curve)
                            ])
                self.wait(0.5)
        self.wait(2)