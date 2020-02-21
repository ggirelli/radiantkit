'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from datetime import datetime
import jinja2 as jj2
import numpy as np  # type: ignore
import os
import plotly.graph_objects as go  # type: ignore
from radiantkit import distance, plot, stat
from radiantkit import series
from typing import Dict, Tuple


class Report(object):
    _env: jj2.Environment
    _template: jj2.Template

    def __init__(self, template: str):
        super(Report, self).__init__()
        self._env = jj2.Environment(
                loader=jj2.PackageLoader('radiantkit', 'templates'),
                autoescape=jj2.select_autoescape(['html', 'xml'])
            )
        self._env.filters['basename'] = os.path.basename
        self._env.filters['dirname'] = os.path.dirname
        self._template = self._env.get_template(template)

    def render(self, path: str, **kwargs) -> None:
        assert "title" in kwargs
        with open(path, "w+") as OH:
            OH.write(self._template.render(**kwargs))


def report_select_nuclei(
        args: argparse.Namespace, opath: str,
        online: bool = False, **kwargs) -> None:
    report = Report('reports/select_nuclei.tpl.html')
    details = kwargs['details']

    figure = plot.plot_nuclear_selection(
        kwargs['data'], args.dna_channel,
        details['size']['range'], details['isum']['range'],
        details['size']['fit'], details['isum']['fit'])

    report.render(
        opath, title="RadIAnT-Kit - Select nuclei",
        online=online, args=args, details=details,
        data=kwargs['data'], series_list=kwargs['series_list'],
        plot_json=figure.to_json(),
        now=str(datetime.now()))


def report_measure_objects(
        args: argparse.Namespace, opath: str,
        online: bool = False, **kwargs) -> None:
    report = Report('reports/measure_objects.tpl.html')

    figure = plot.plot_nuclear_features(
        kwargs['data'], kwargs['spx_data'],
        kwargs['series_list'].particle_feature_labels())

    report.render(
        opath, title="RadIAnT-Kit - Measure objects",
        online=online, args=args,
        data=kwargs['data'], series_list=kwargs['series_list'],
        plot_json=figure.to_json(),
        now=str(datetime.now()))


def report_radial_population(
        args: argparse.Namespace, opath: str,
        profile_data: series.RadialProfileData,
        online: bool = False, **kwargs) -> None:
    report = Report('reports/radial_population.tpl.html')

    roots: Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}
    distance_type_set = set()
    figures: Dict[str, Dict[str, go.Figure]] = {}
    for channel_name in profile_data:
        figures[channel_name] = {}
        roots[channel_name] = {}
        for distance_type in profile_data[channel_name]:
            distance_type_set.add(distance_type)
            profile, raw_data = profile_data[channel_name][distance_type]

            roots_dict = {}
            for stat_name in profile:
                roots_dict[stat_name] = stat.get_radial_profile_roots(
                    profile[stat_name])
            roots[channel_name][distance_type] = roots_dict

            figures[channel_name][distance_type] = plot.plot_profile(
                distance_type, profile, raw_data, roots_dict)

    report.render(
        opath, title="RadIAnT-Kit - Population radiality",
        online=online, args=args, series_list=kwargs['series_list'],
        profiles=profile_data, figures=figures, roots=roots,
        dtypes=distance_type_set, dlabs=distance.__distance_labels__,
        now=str(datetime.now()))
