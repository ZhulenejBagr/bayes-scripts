import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from flow123d_simulation import generate_time_axis


class MeasuredData:
    def __init__(self, config):
        self.borehole_names = []
        self.measured_data = {}
        self.interp_data = {}
        self.temp_color = {}
        self._config = config

    def initialize(self):
        self.borehole_names, self.measured_data = self.read_chandler_data()

        for bname in self.borehole_names:
            t = self.measured_data[bname]["time"]
            p = self.measured_data[bname]["pressure"]
            self.interp_data[bname] = interpolate.CubicSpline(t, p, bc_type='natural')

        self.temp_color = {'HGT1-1': 'sienna', 'HGT1-2': 'yellow', 'HGT1-3': 'orange', 'HGT1-4': 'green',
                           'HGT1-5': 'red', 'HGT2-1': 'teal', 'HGT2-2': 'cyan', 'HGT2-3': 'blue', 'HGT2-4': 'violet'}
        
        self.bnames_dict = {'V1':'HGT1-5', 'V2':'HGT1-4',
                            'H1':'HGT2-4', 'H2':'HGT2-3'}

    def generate_measured_samples(self, boreholes, cond_boreholes):
        times = np.array(generate_time_axis(self._config))

        # sample measured data at generated times
        values = []
        for bname in boreholes:
            bn = self.bnames_dict[bname]
            p = self.interp_data[bn](times)
            values.extend(p)

        values.extend(self.conductivity_measurement(cond_boreholes, times))
        return times, values

    def conductivity_measurement(self, boreholes, times):
        measurements = {"V1_cond": 2e-17, "V2_cond": 1e-19, "H1_cond": 3e-19, "H2_cond": 7e-21}
        n = len(times)
        # initial bulk conductivity
        # init_cond = np.log10(1e7 * 6e-22)
        # values at points V01_cond, V02_cond, H01_cond, H02_cond
        # values = np.log10(1e7 * np.array([2e-17, 1e-19, 3e-19, 7e-21]))
        values = []
        for b in boreholes:
            values.append(measurements[b])
        values = np.log10(1e7 * np.array(values))
        # make time rows
        # values = np.tile(values, (n, 1)).transpose()

        # replace initial conductivity
        # bored_time = float(self._config['bored_time'])
        # values[:, times < bored_time] = init_cond * np.ones(np.shape(values[:, times < bored_time]))
        return values.flatten()

    def plot_interp_data(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time [d]')
        ax1.set_ylabel('pressure head [m]')

        self.plot_data_set(self.borehole_names, self.measured_data, ax1, linestyle='solid')

        for bname in self.borehole_names:
            t = self.measured_data[bname]["time"]
            # p = interpolate.splev(t, self.interp_data[bname])
            p = self.interp_data[bname](t)
            ax1.plot(t, p, color='black', label=bname, linestyle='dotted')

        self.additional_annotation(ax1)

        ax1.tick_params(axis='y')
        ax1.legend(ncol=3)

        fig.tight_layout()
        # plt.show()

        fig_file = os.path.join(self._config["work_dir"], "interp_data_TSX.pdf")
        plt.savefig(fig_file)

    def plot_all_data(self):
        bnames_ch, data_ch = self.read_chandler_data()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time [d]')
        ax1.set_ylabel('pressure head [m]')
        self.plot_data_set(bnames_ch, data_ch, ax1, linestyle='solid')

        self.additional_annotation(ax1)

        props = dict(boxstyle='square', facecolor='white', alpha=0.25)

        ax1.tick_params(axis='y')
        ax1.legend(ncol=3)

        fig.tight_layout()
        # plt.show()
        fig_file = os.path.join(self._config["work_dir"], "measured_data_TSX.pdf")
        plt.savefig(fig_file)

    def additional_annotation(self, axis):
        bored_time = int(self._config["bored_time"])
        axis.axvline(x=bored_time, ymin=0.0, ymax=1.0, color='k', linewidth=0.25)
        # axis.set_xticks(list(ax1.get_xticks()) + [17])
        axis.axhline(y=300, xmin=0.0, xmax=1.0, color='gray', linewidth=0.1)
        axis.axhline(y=275, xmin=0.0, xmax=1.0, color='gray', linewidth=0.1)
        axis.axhline(y=250, xmin=0.0, xmax=1.0, color='gray', linewidth=0.1)
        axis.text(18, 11, str(bored_time), fontsize=5)
        axis.text(-15, 278, '275', fontsize=5)
        axis.text(-15, 253, '250', fontsize=5)

    def plot_comparison(self, computed_data, output_dir, boreholes):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time [d]')
        ax1.set_ylabel('pressure [m]')

        # self.plot_data_set(self.borehole_names, self.measured_data, ax1, linestyle='solid')
        t = generate_time_axis(self._config)

        idx = 0
        label_comsol="comsol"
        for bname in boreholes:
            bn = self.bnames_dict[bname]
            p_interp = self.interp_data[bn](t)
            end_idx = idx + len(t)
            p_comp = computed_data[idx:end_idx]
            ax1.plot(t, p_interp, color=self.temp_color[bname], label="d:"+bname, linestyle='dotted')
            ax1.plot(t, p_comp, color=self.temp_color[bname], label="m:"+bname, linestyle='solid')
            idx = idx + len(t)

        # sorting legend handles and labels
        import operator
        handles, labels = ax1.get_legend_handles_labels()
        hl = sorted(zip(handles, labels), key=operator.itemgetter(1), reverse=True)
        handles2, labels2 = zip(*hl)

        ax1.tick_params(axis='y')
        ax1.legend(handles2, labels2, ncol=3)

        fig.tight_layout()
        # plt.show()

        fig_file = os.path.join(output_dir, "observe_comparison.pdf")
        plt.savefig(fig_file)

    def read_csv_graph_data(self, csv_file):
        with open(csv_file, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            borehole_names = []
            data = {}
            for row in csv_reader:
                if line_count == 0:
                    for col in row:
                        if col != "":
                            borehole_names.append(col)
                            data[col] = {"time": [], "pressure": []}

                if line_count < 2:
                    line_count = line_count + 1
                    continue
                assert 2*len(borehole_names) == len(row)
                for i in range(0, len(borehole_names)):
                    try:
                        t = float(row[2*i])
                        v = float(row[2*i+1])
                        d = data[borehole_names[i]]
                        d["time"].append(t)
                        d["pressure"].append(v)
                    except:
                        line_count = line_count + 1
                        continue
                line_count = line_count + 1
        return borehole_names, data

    def read_chandler_data(self):
        # read Chandler data
        datafile1 = os.path.join(self._config["measured_data_dir"], "chandler_wpd_datasets_HGT1.csv")
        datafile2 = os.path.join(self._config["measured_data_dir"], "chandler_wpd_datasets_HGT2.csv")
        borehole_names, data = self.read_csv_graph_data(datafile1)
        borehole_names_2, data_2 = self.read_csv_graph_data(datafile2)

        borehole_names.extend(borehole_names_2)
        data.update(data_2)
        excavation_start = 11.4 #12.7
        # sorting and cropping data
        for bname, dat in data.items():
            t = np.array(dat["time"])
            v = np.array(dat["pressure"])
            permutation = np.argsort(t)
            start_idx = np.searchsorted(t, excavation_start, sorter=permutation)
            t = t - excavation_start
            dat["time"] = t[permutation][start_idx:]
            # transform pressure from [kPa] to pressure head [m]
            # 1 kPa = [/rho/g] = 0.1 m
            dat["pressure"] = v[permutation][start_idx:] / 10

        return borehole_names, data

    def plot_data_set(self, bnames, data, axes, linestyle):
        for bname in bnames:
            bn = bname if bname in self.temp_color.keys() else self.bnames_dict[bname]
            color = self.temp_color[bn]
            axes.plot(data[bname]["time"], data[bname]["pressure"], color=color,
                      label=bname, linestyle=linestyle)


if __name__ == "__main__":

    from bp_simunek.simulation.flow_wrapper import Wrapper

    config_dict = Wrapper.setup_config()

    os.makedirs(config_dict["work_dir"], mode=0o775, exist_ok=True)
    os.chdir(config_dict["work_dir"])

    md = MeasuredData(config_dict)
    md.initialize()

    md.plot_all_data()
    md.plot_interp_data()

    boreholes = ["HGT1-5", "HGT1-4", "HGT2-4", "HGT2-3"]
    times, values = md.generate_measured_samples(boreholes)

    print(times, values)
