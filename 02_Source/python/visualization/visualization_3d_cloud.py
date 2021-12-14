import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d as a3
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import math

cloud = np.array([[1.695595,-0.200348,-0.743032, 153.994720, 6.093055, 1.695595, 1.691803, -0.200208, -0.741456, 1],[1.367459,0.250888,0.366419, 87.636490, 161.450348, 1.367459, 1.365946, 0.250678, 0.366050, 1],[1.427906,0.146808,0.169010, 105.381104, 128.539658, 1.427906, 1.427568, 0.146832, 0.168816, 1],[1.361322,0.280790,0.009545, 82.622162, 104.042480, 1.361322, 1.360412, 0.280658, 0.009584, 1],[1.663774,-0.421559,0.578370, 183.742508, 178.977509, 1.663774, 1.675695, -0.424549, 0.582695, 1],[1.774876,-0.698781,0.605505, 214.615479, 177.553726, 1.774876, 1.795467, -0.707111, 0.610053, 1],[1.668039,-0.132773,-0.670250, 145.511658, 14.099774, 1.668039, 1.672999, -0.151532, -0.668832, 1],[1.447411,0.423224,0.211958, 63.671783, 134.716721, 1.447411, 1.457361, 0.426124, 0.213490, 1],[1.244821,0.365885,0.386428, 63.336342, 170.794327, 1.244821, 1.686909, -0.134227, -0.677831, 0],[1.668039,-0.132773,-0.670250, 145.511658, 14.099774, 1.668039, 1.672999, -0.151532, -0.668832, 1],[1.427906,0.146808,0.169010, 105.381104, 128.539658, 1.427906, 1.427568, 0.146832, 0.168816, 1],[1.270800,-0.176060,0.536857, 158.479385, 195.440231, 1.270800, 1.265297, -0.185617, 0.526042, 1],[1.472264,0.021076,0.628187, 124.850632, 196.369766, 1.472264, 1.433497, 0.020674, 0.611167, 1],[1.686640,-0.483589,0.528232, 191.077744, 171.400940, 1.686640, 1.702208, -0.488380, 0.532611, 1],[1.342550,0.314383,-0.014738, 76.482933, 100.084915, 1.342550, 1.346628, 0.315375, -0.014858, 1],[1.776581,-0.915968,0.531175, 241.427399, 168.277237, 1.776581, 1.782620, -0.908026, 0.513346, 1],[1.473657,0.167417,0.229660, 103.006546, 136.785599, 1.473657, 1.473763, 0.167469, 0.229891, 1],[1.463384,0.202562,0.096357, 97.547462, 116.986031, 1.463384, 1.475778, 0.204059, 0.097237, 1],[1.551724,-0.000508,-0.657165, 128.072021, 9.328538, 1.551724, 1.555025, -0.000290, -0.658570, 1],[1.615752,-0.128550,0.222503, 145.503296, 132.795929, 1.615752, 1.595465, -0.127714, 0.219994, 1],[1.349509,0.296425,0.334795, 79.676208, 157.079102, 1.349509, 1.339247, 0.294103, 0.332231, 1],[1.423796,0.182962,0.233507, 99.729317, 138.580719, 1.423796, 1.460046, 0.188081, 0.239635, 1],[1.649102,-0.294517,0.196288, 167.290253, 128.685989, 1.649102, 1.664945, -0.297609, 0.198078, 1],[1.647406,-0.103244,-0.632066, 141.787506, 18.091818, 1.647406, 1.675770, -0.104741, -0.643069, 1],[2.049659,-1.115318,0.100877, 247.712570, 113.327629, 2.049659, 2.034611, -1.107938, 0.100142, 1],[1.358657,-0.091307,0.528542, 142.784805, 188.083923, 1.358657, 1.345016, -0.084091, 0.558291, 1],[1.705331,-0.184680,-0.732837, 151.825043, 7.958707, 1.705331, 1.700665, -0.184322, -0.731395, 1],[1.592984,-0.271079,0.537427, 165.437576, 176.721680, 1.592984, 1.612847, -0.274152, 0.544186, 1],[1.669738,-0.327439,-0.598237, 171.142380, 23.678003, 1.669738, 1.710158, -0.335153, -0.612804, 1],[1.587092,0.440351,0.268111, 66.959213, 139.665085, 1.587092, 1.525696, 0.423006, 0.258145, 1],[1.519063,0.415683,0.282876, 67.798271, 143.467865, 1.519063, 1.525696, 0.423006, 0.258145, 1],[1.632597,-0.670349,0.152148, 218.332672, 123.002602, 1.632597, 1.654124, -0.658373, 0.175274, 1],[1.893525,-0.922435,0.049264, 235.173431, 108.223755, 1.893525, 1.895923, -0.945842, 0.025314, 1],[1.792869,-0.888428,0.139901, 237.017532, 119.666985, 1.792869, 1.853229, -0.918646, 0.144899, 1],[1.774845,-0.926837,0.535313, 242.885544, 168.854462, 1.774845, 1.782620, -0.908026, 0.513346, 1],[1.364907,0.373220,0.362774, 67.843231, 160.973083, 1.364907, 1.379748, 0.362754, 0.338137, 1],[1.550170,-0.178592,0.536768, 153.345795, 178.678146, 1.550170, 1.554491, -0.179084, 0.538236, 1],[1.647406,-0.103244,-0.632066, 141.787506, 18.091818, 1.647406, 1.675770, -0.104741, -0.643069, 1],[1.358657,-0.091307,0.528542, 142.784805, 188.083923, 1.358657, 1.345016, -0.084091, 0.558291, 1],[1.342550,0.314383,-0.014738, 76.482933, 100.084915, 1.342550, 1.346628, 0.315375, -0.014858, 1],[1.705331,-0.184680,-0.732837, 151.825043, 7.958707, 1.705331, 1.700665, -0.184322, -0.731395, 1],[1.550170,-0.178592,0.536768, 153.345795, 178.678146, 1.550170, 1.554491, -0.179084, 0.538236, 1],[2.006620,-1.068442,-0.424229, 245.140808, 55.988808, 2.006620, 2.051143, -1.058618, -0.472497, 1],[2.059062,-1.045992,-0.492387, 239.758804, 49.891048, 2.059062, 2.084862, -1.060574, -0.499222, 1],[1.267334,0.675129,-0.430651, 10.802470, 27.742165, 1.267334, 1.257980, 0.672492, -0.427021, 1],[1.740273,-0.413278,-0.628383, 180.245346, 23.061773, 1.740273, 1.734986, -0.412214, -0.626726, 1],[1.667176,-0.345472,-0.577623, 173.588394, 26.277143, 1.667176, 1.675366, -0.355427, -0.538160, 1],[1.586946,-0.077280,-0.586345, 138.713440, 21.214350, 1.586946, 1.575449, -0.076700, -0.582261, 1],[1.851622,-0.992514,0.114948, 245.925262, 116.157547, 1.851622, 1.880132, -1.007068, 0.117913, 1],[1.406819,0.779345,-0.449010, 6.125016, 32.283272, 1.406819, 1.721782, -0.356642, -0.596536, 0],[1.811642,-0.691911,-0.412432, 212.023438, 52.415600, 1.811642, 1.842219, -0.701203, -0.429715, 1],[1.967330,-0.957398,-0.427478, 235.062714, 54.696587, 1.967330, 1.973249, -0.931798, -0.463048, 1],[1.370373,0.387582,-0.425921, 65.777458, 34.122471, 1.370373, 1.366265, 0.386622, -0.424633, 1],[1.874455,-0.934760,0.026263, 237.710373, 105.582451, 1.874455, 1.895923, -0.945842, 0.025314, 1],[1.717697,-0.455075,0.163695, 186.285278, 123.465797, 1.717697, 1.708116, -0.452254, 0.162403, 1],[1.599155,-0.509787,0.147313, 198.132706, 122.766296, 1.599155, 1.587440, -0.506247, 0.146156, 1],[1.739192,-0.069599,-0.163163, 136.803940, 81.860596, 1.739192, 1.740258, -0.067169, -0.163602, 1],[1.707609,-0.586286,0.118051, 203.534225, 117.709106, 1.707609, 1.754987, -0.602552, 0.121939, 1],[1.656075,-0.613683,0.119922, 209.524185, 118.430992, 1.656075, 1.670586, -0.619167, 0.121346, 1],[1.366843,0.280665,0.024562, 82.825562, 106.453438, 1.366843, 1.358559, 0.279132, 0.024519, 1],[1.703222,-0.205163,-0.723327, 154.500259, 9.070076, 1.703222, 1.697995, -0.202997, -0.721282, 1],[1.893098,-0.958292,0.015202, 239.364716, 104.266640, 1.893098, 1.895923, -0.945842, 0.025314, 1],[1.716872,-0.002011,-0.111582, 128.257721, 88.201942, 1.716872, 1.977445, -1.038236, 0.495506, 0],[1.335147,0.313102,0.308110, 76.408279, 153.269150, 1.335147, 1.322195, 0.322158, 0.298162, 1],[1.340764,0.341682,0.018841, 71.934921, 105.591591, 1.340764, 1.329660, 0.338794, 0.019054, 1],[1.355409,0.291373,0.036107, 80.706421, 108.360703, 1.355409, 1.355609, 0.291419, 0.036402, 1],[1.387008,0.356190,0.443149, 71.502975, 172.790024, 1.387008, 1.390666, 0.356393, 0.443150, 1],[1.425881,0.280062,0.127058, 84.789062, 122.103851, 1.425881, 1.383767, 0.268266, 0.119489, 1],[1.053652,0.593364,0.063517, 4.107100, 115.762184, 1.053652, 1.026793, 0.578193, 0.061936, 1],[1.669763,-0.664847,0.176813, 215.597046, 125.796074, 1.669763, 1.654124, -0.658373, 0.175274, 1],[1.728743,-0.498087,0.118148, 191.386658, 117.535492, 1.728743, 1.675188, -0.483435, 0.114719, 1],[1.553449,-0.253199,0.220624, 163.858124, 133.744827, 1.553449, 1.592957, -0.262085, 0.227172, 1],[1.707609,-0.574488,0.105255, 202.014267, 116.060532, 1.707609, 1.754987, -0.602552, 0.121939, 1],[1.340907,0.338105,-0.037123, 72.527687, 96.409355, 1.340907, 1.340583, 0.338050, -0.037183, 1],[1.768649,-0.876135,0.161014, 236.981339, 122.528374, 1.768649, 1.720454, -0.852382, 0.156943, 1],[1.726220,-0.354178,0.235897, 173.138550, 132.564163, 1.726220, 1.729803, -0.352940, 0.199641, 1],[1.755588,-0.894315,0.506149, 240.070328, 165.927673, 1.755588, 1.782620, -0.908026, 0.513346, 1],[1.655813,-0.362167,0.508389, 176.119431, 170.047256, 1.655813, 1.647774, -0.360474, 0.506049, 1],[1.538980,0.417442,0.134778, 68.325920, 121.766716, 1.538980, 1.690775, -0.871971, 0.505957, 0],[1.758221,-0.896602,0.629716, 240.188599, 181.294083, 1.758221, 1.726605, -0.872126, 0.628985, 1],[1.271532,0.282775,0.151355, 79.074471, 128.687332, 1.271532, 1.272407, 0.276064, 0.156419, 1],[1.268490,0.276275,0.227904, 80.084435, 142.026474, 1.268490, 1.268046, 0.276260, 0.227827, 1],[1.370501,0.270322,0.207617, 84.606506, 135.827774, 1.370501, 1.334048, 0.263288, 0.202108, 1],[1.327675,0.240049,0.149446, 88.223145, 127.263756, 1.327675, 1.315382, 0.237836, 0.148012, 1],[1.971048,-1.036045,0.494420, 243.638931, 157.685043, 1.971048, 1.980431, -1.040296, 0.496238, 1],[1.605869,0.115653,0.200600, 112.155785, 129.981705, 1.605869, 1.339240, 0.337754, -0.037114, 0],[1.467311,-0.023354,0.140822, 131.501602, 123.613976, 1.467311, 1.487008, -0.024042, 0.143204, 1],[1.827057,-0.455137,0.233170, 182.804031, 130.576462, 1.827057, 1.021581, 0.575378, 0.061609, 0],[1.215242,-0.046992,0.543122, 136.507141, 200.823410, 1.215242, 1.210952, -0.042092, 0.549623, 1],[1.341100,0.302842,0.578065, 78.320496, 197.328415, 1.341100, 1.328171, 0.299992, 0.572329, 1],[1.622068,-0.563697,0.188326, 204.453873, 128.042572, 1.622068, 1.617703, -0.555280, 0.157336, 1],[1.324337,0.339635,0.198200, 71.579636, 135.425125, 1.324337, 1.334668, 0.340157, 0.173201, 1],[1.681292,-0.440811,0.501085, 185.680893, 168.067810, 1.681292, 1.683465, -0.441874, 0.501566, 1],[1.323275,0.056262,0.180254, 118.646278, 132.467957, 1.323275, 1.315378, 0.055459, 0.180327, 1],[1.697352,-0.521318,0.492456, 195.569977, 166.329010, 1.697352, 1.711915, -0.526277, 0.496904, 1],[1.451535,0.380562,0.287508, 70.320610, 146.075714, 1.451535, 1.450064, 0.396564, 0.270121, 1],[1.368612,0.361798,0.343790, 69.842163, 157.763077, 1.368612, 1.379748, 0.362754, 0.338137, 1],[1.553295,0.019636,0.184763, 125.218903, 128.668793, 1.553295, 1.551128, 0.020124, 0.184530, 1],[1.573109,-0.097455,0.176642, 141.629120, 127.203499, 1.573109, 1.575083, -0.084508, 0.167489, 1],[1.712775,-0.479878,0.636130, 189.638611, 184.208679, 1.712775, 1.709298, -0.490291, 0.654285, 1],[1.271915,-0.186924,0.528017, 160.331802, 193.829758, 1.271915, 1.265297, -0.185617, 0.526042, 1],[1.235301,-0.097679,0.534240, 145.396027, 197.645096, 1.235301, 1.233378, -0.094154, 0.534466, 1],[1.506969,0.231224,0.219081, 94.244011, 134.483322, 1.506969, 1.471538, 0.225953, 0.214265, 1],[1.280807,-0.194007,0.524907, 161.323959, 192.661545, 1.280807, 1.282358, -0.197045, 0.522818, 1],[1.592929,-0.244352,0.519176, 161.747498, 174.203629, 1.592929, 1.576953, -0.241806, 0.514058, 1],[1.386057,0.364434,0.339733, 70.155670, 156.423706, 1.386057, 1.379748, 0.362754, 0.338137, 1],[1.836333,-0.898307,0.692446, 235.620758, 185.457764, 1.836333, 1.867139, -0.912196, 0.707576, 1],[1.327775,0.323696,0.299537, 74.366531, 152.130569, 1.327775, 1.322195, 0.322158, 0.298162, 1],[1.274477,0.278866,0.083657, 79.862213, 116.940811, 1.274477, 1.284742, 0.280955, 0.084094, 1],[1.316808,0.260308,0.259451, 84.510162, 145.846619, 1.316808, 1.309569, 0.263418, 0.257074, 1],[1.485912,0.221503,0.078072, 95.204803, 114.059181, 1.485912, 1.475778, 0.204059, 0.097237, 1],[1.404253,0.212524,0.265439, 94.704483, 144.085480, 1.404253, 1.425971, 0.216315, 0.267655, 1],[1.430007,0.065342,0.141589, 117.947479, 124.282806, 1.430007, 1.403207, 0.064449, 0.138988, 1],[1.394592,-0.052865,0.542740, 136.339569, 188.118362, 1.394592, 1.423942, -0.039440, 0.543187, 1],[1.505360,-0.129382,0.583468, 146.908447, 187.770584, 1.505360, 1.507097, -0.129985, 0.584186, 1],[1.676817,-0.491965,0.727947, 192.546234, 198.007294, 1.676817, 1.645184, -0.482789, 0.714026, 1],[1.470779,0.057230,0.589790, 119.439560, 190.721115, 1.470779, 1.449690, 0.062520, 0.579119, 1],[1.644970,-0.315149,0.655641, 170.148407, 190.186157, 1.644970, 1.638317, -0.311873, 0.649230, 1],[1.701818,-0.537728,0.717732, 197.514038, 195.283722, 1.701818, 1.690648, -0.552241, 0.726480, 1],[1.474903,0.000987,0.533064, 127.852760, 182.013092, 1.474903, 1.466297, 0.000845, 0.530022, 1],[1.487936,-0.048456,0.522951, 135.164490, 179.821304, 1.487936, 1.490332, -0.048134, 0.523733, 1],[1.547242,-0.192195,0.560145, 155.327911, 182.146210, 1.547242, 1.557281, -0.193525, 0.563815, 1],[1.451654,0.049316,0.537415, 120.526085, 183.945908, 1.451654, 1.447092, 0.049244, 0.535769, 1],[1.424689,-0.039742,0.543385, 134.136902, 186.409332, 1.424689, 1.423942, -0.039440, 0.543187, 1],[1.503138,-0.172008,0.587427, 153.175232, 188.476059, 1.503138, 1.484400, -0.169096, 0.580210, 1],[1.453020,0.062553,0.580381, 118.528877, 190.374817, 1.453020, 1.449690, 0.062520, 0.579119, 1],[1.707609,-0.586286,0.118051, 203.534225, 117.709106, 1.707609, 1.754987, -0.602552, 0.121939, 1],[1.703222,-0.205163,-0.723327, 154.500259, 9.070076, 1.703222, 1.697995, -0.202997, -0.721282, 1],[1.971048,-1.036045,0.494420, 243.638931, 157.685043, 1.971048, 1.980431, -1.040296, 0.496238, 1],[1.335147,0.313102,0.308110, 76.408279, 153.269150, 1.335147, 1.322195, 0.322158, 0.298162, 1],[1.274477,0.278866,0.083657, 79.862213, 116.940811, 1.274477, 1.284742, 0.280955, 0.084094, 1],[1.370501,0.270322,0.207617, 84.606506, 135.827774, 1.370501, 1.334048, 0.263288, 0.202108, 1],[1.268490,0.276275,0.227904, 80.084435, 142.026474, 1.268490, 1.268046, 0.276260, 0.227827, 1],[1.425881,0.280062,0.127058, 84.789062, 122.103851, 1.425881, 1.383767, 0.268266, 0.119489, 1],[1.280807,-0.194007,0.524907, 161.323959, 192.661545, 1.280807, 1.282358, -0.197045, 0.522818, 1],[1.586946,-0.077280,-0.586345, 138.713440, 21.214350, 1.586946, 1.575449, -0.076700, -0.582261, 1],[1.271915,-0.186924,0.528017, 160.331802, 193.829758, 1.271915, 1.265297, -0.185617, 0.526042, 1],[1.340907,0.338105,-0.037123, 72.527687, 96.409355, 1.340907, 1.340583, 0.338050, -0.037183, 1],[1.553295,0.019636,0.184763, 125.218903, 128.668793, 1.553295, 1.551128, 0.020124, 0.184530, 1],[1.053652,0.593364,0.063517, 4.107100, 115.762184, 1.053652, 1.026793, 0.578193, 0.061936, 1],[1.592929,-0.244352,0.519176, 161.747498, 174.203629, 1.592929, 1.576953, -0.241806, 0.514058, 1],[1.505360,-0.129382,0.583468, 146.908447, 187.770584, 1.505360, 1.507097, -0.129985, 0.584186, 1],[1.470779,0.057230,0.589790, 119.439560, 190.721115, 1.470779, 1.449690, 0.062520, 0.579119, 1],[1.503138,-0.172008,0.587427, 153.175232, 188.476059, 1.503138, 1.484400, -0.169096, 0.580210, 1],[1.701818,-0.537728,0.717732, 197.514038, 195.283722, 1.701818, 1.690648, -0.552241, 0.726480, 1],[1.453020,0.062553,0.580381, 118.528877, 190.374817, 1.453020, 1.449690, 0.062520, 0.579119, 1],[1.547242,-0.192195,0.560145, 155.327911, 182.146210, 1.547242, 1.557281, -0.193525, 0.563815, 1],[2.076454,-1.066022,-0.775349, 240.944855, 20.351912, 2.076454, 2.109412, -1.047273, -0.808385, 1],[2.039435,-0.982235,-0.772238, 233.956635, 19.196337, 2.039435, 2.032577, -0.978769, -0.769305, 1],[1.276237,0.598190,-0.415076, 24.882975, 30.948526, 1.276237, 1.292997, 0.605773, -0.420243, 1],[1.353308,0.704136,-0.431745, 13.532350, 32.313499, 1.353308, 1.313970, 0.681853, -0.420001, 1],[1.295709,0.626690,-0.407772, 21.593575, 33.263947, 1.295709, 1.299033, 0.628642, -0.408571, 1],[1.745355,-0.465322,-0.448152, 186.653351, 46.010990, 1.745355, 1.734704, -0.461942, -0.444918, 1],[2.071572,-1.065279,-0.478204, 241.132202, 51.714935, 2.071572, 2.084862, -1.060574, -0.499222, 1],[2.030078,-1.077629,-0.445101, 244.782837, 54.264355, 2.030078, 2.051143, -1.058618, -0.472497, 1],[1.259334,0.714246,-0.426185, 3.224365, 28.047342, 1.259334, 1.242499, 0.704397, -0.421030, 1],[1.289402,0.715351,-0.418050, 5.945496, 31.171598, 1.289402, 1.265595, 0.701398, -0.411259, 1],[1.266146,0.636764,-0.417920, 17.358650, 29.883999, 1.266146, 1.263564, 0.634879, -0.417448, 1],[1.666923,-0.151200,-0.666964, 147.955338, 14.474221, 1.666923, 1.672999, -0.151532, -0.668832, 1],[1.435214,0.379062,-0.420699, 69.894630, 38.012173, 1.435214, 1.412308, 0.411617, -0.414094, 1],[1.666588,-0.157132,-0.735578, 148.742355, 5.399086, 1.666588, 1.694367, -0.165788, -0.742275, 1],[2.115401,-1.050345,-0.812209, 237.235031, 18.030916, 2.115401, 2.109412, -1.047273, -0.808385, 1],[2.620964,-1.085269,-1.106476, 219.095947, 9.624015, 2.620964, 1.550522, -0.043588, -0.575935, 0],[1.718434,-0.135015,-0.770117, 145.285095, 3.906953, 1.718434, 1.714121, -0.132916, -0.744095, 1],[1.548042,-0.043490,-0.574978, 134.180511, 20.787035, 1.548042, 1.553420, -0.043611, -0.577062, 1],[1.589673,-0.058580,-0.612486, 136.107010, 17.736080, 1.589673, 1.597753, -0.059065, -0.615572, 1],[1.556075,-0.064941,-0.563768, 137.181458, 22.793770, 1.556075, 1.541534, -0.064263, -0.558411, 1],[1.785634,-0.263349,-0.792916, 160.446075, 4.808312, 1.785634, 1.796263, -0.266385, -0.809771, 1],[1.769388,-0.327589,-0.784268, 168.731323, 4.986675, 1.769388, 1.752387, -0.323973, -0.776784, 1],[1.774991,-0.402699,-0.794441, 177.912308, 4.033561, 1.774991, 1.779964, -0.357007, -0.756994, 1],[1.781829,-0.338989,-0.669439, 169.854507, 19.845274, 1.781829, 1.740491, -0.330767, -0.653869, 1],[1.660536,-0.324542,-0.418733, 170.997711, 47.023121, 1.660536, 1.662540, -0.326344, -0.418573, 1],[1.674983,-0.131509,-0.726843, 145.273026, 7.033005, 1.674983, 1.671704, -0.123949, -0.702153, 1],[1.646646,-0.096546,-0.689743, 140.899063, 10.346992, 1.646646, 1.649586, -0.097584, -0.691118, 1],[1.659404,-0.123057,-0.696895, 144.314636, 10.107247, 1.659404, 1.671704, -0.123949, -0.702153, 1],[1.896418,-0.750004,-0.751382, 215.006638, 15.333565, 1.896418, 1.339850, 0.492065, -0.423022, 0],[1.689150,-0.192856,-0.702971, 153.118164, 10.942925, 1.689150, 1.698665, -0.193795, -0.705743, 1],[1.304941,0.549695,-0.419013, 35.326965, 31.858540, 1.304941, 1.299760, 0.546596, -0.417374, 1],[1.771897,-0.336137,-0.734818, 169.735062, 11.264431, 1.771897, 1.756398, -0.334089, -0.728430, 1],[1.331042,0.488870,-0.420229, 47.197548, 33.042835, 1.331042, 1.336780, 0.490959, -0.422019, 1],[1.758805,-0.352188,-0.746067, 172.053406, 9.178248, 1.758805, 1.779964, -0.357007, -0.756994, 1],[1.608980,-0.153851,-0.420107, 149.036484, 45.057751, 1.608980, 1.594769, -0.152977, -0.416984, 1],[1.696850,-0.179029,-0.676502, 151.211502, 14.790197, 1.696850, 1.671599, -0.176775, -0.665498, 1],[1.772651,-0.258225,-0.728516, 160.047806, 12.085463, 1.772651, 1.736574, -0.253305, -0.712822, 1],[1.311452,0.571772,-0.408396, 32.083584, 33.990440, 1.311452, 1.299760, 0.546596, -0.417374, 1],[1.340855,0.512512,-0.410630, 43.909901, 35.126209, 1.340855, 1.328632, 0.508143, -0.406773, 1],[1.367033,0.421568,-0.425676, 60.156090, 33.994896, 1.367033, 1.362226, 0.420323, -0.424124, 1],[1.376839,0.417825,-0.410071, 61.237339, 36.976303, 1.376839, 1.362226, 0.420323, -0.424124, 1],[1.407765,0.410752,-0.412814, 63.809246, 37.987049, 1.407765, 1.412308, 0.411617, -0.414094, 1],[1.369657,0.384490,-0.414568, 66.241631, 35.910370, 1.369657, 1.368879, 0.384329, -0.414384, 1],[1.404270,0.320258,-0.411157, 77.826767, 38.086082, 1.404270, 1.412262, 0.322220, -0.413390, 1],[1.402228,0.300667,-0.422816, 80.827339, 36.163078, 1.402228, 1.408919, 0.302062, -0.424831, 1],[1.431857,0.297265,-0.413798, 82.326164, 38.921288, 1.431857, 1.417446, 0.294457, -0.409468, 1],[1.721436,-0.374474,-0.583070, 175.857819, 27.983461, 1.721436, 1.719229, -0.373858, -0.582168, 1],[1.904222,-0.977094,0.006717, 240.886398, 103.276001, 1.904222, 1.915474, -0.969252, 0.015544, 1],[1.581286,-0.050846,-0.132344, 135.074112, 84.087311, 1.581286, 1.581286, -0.055919, -0.132528, 1],[1.846538,-0.681043,-0.451675, 209.140732, 48.686573, 1.846538, 1.833780, -0.674612, -0.448555, 1],[1.875157,-0.813559,-0.428718, 223.449570, 52.201340, 1.875157, 1.889180, -0.816312, -0.432355, 1],[1.975474,-0.956623,-0.443003, 234.534943, 53.164654, 1.975474, 1.973249, -0.931798, -0.463048, 1],[1.630850,-0.189592,-0.120712, 153.575790, 86.216026, 1.630850, 1.686913, -0.197638, -0.125352, 1],[1.857926,-0.707228,-0.433338, 211.744003, 51.187706, 1.857926, 1.842219, -0.701203, -0.429715, 1],[1.922267,-0.814545,-0.460695, 221.223236, 49.774261, 1.922267, 1.930490, -0.818110, -0.462406, 1],[1.978500,-0.939846,-0.465025, 232.506546, 50.791359, 1.978500, 1.973249, -0.931798, -0.463048, 1],[1.642521,-0.130664,-0.138869, 145.501190, 83.899864, 1.642521, 1.724263, -0.579872, 0.105817, 0],[1.341343,0.347663,-0.046838, 70.978127, 94.817924, 1.341343, 1.337511, 0.346852, -0.047136, 1],[1.696702,-0.495654,-0.074416, 192.268127, 92.850960, 1.696702, 1.827877, -0.938979, 0.051293, 0],[1.760562,-0.521049,-0.110326, 193.110367, 88.713707, 1.760562, 1.520347, -0.060911, 0.138686, 0],[1.579490,-0.019126,0.118080, 130.663940, 118.946838, 1.579490, 1.531065, -0.037300, 0.108264, 1],[1.342094,0.230731,0.077206, 90.177887, 115.155815, 1.342094, 1.335425, 0.229502, 0.076867, 1],[1.015921,0.558174,0.064047, 7.126237, 116.369415, 1.015921, 1.013424, 0.557468, 0.063637, 1],[1.808319,-0.928394,0.053443, 240.948395, 109.001915, 1.808319, 1.807005, -0.880327, 0.046859, 1],[1.031062,0.562236,0.082098, 8.034394, 120.017502, 1.031062, 1.013424, 0.557468, 0.063637, 1],[1.335589,0.320963,0.037881, 75.130463, 108.739861, 1.335589, 1.342197, 0.322515, 0.038054, 1],[1.574643,-0.083764,0.160335, 139.703033, 124.901077, 1.574643, 1.575083, -0.084508, 0.167489, 1],[1.342094,0.226825,0.093561, 90.818115, 117.836861, 1.342094, 1.335425, 0.225775, 0.093101, 1],[1.331221,0.343285,0.043339, 71.268066, 109.662354, 1.331221, 1.339167, 0.345406, 0.043478, 1],[1.373900,0.261131,0.047083, 86.185654, 110.039284, 1.373900, 1.375267, 0.261430, 0.047137, 1],[1.025716,0.555322,0.050199, 8.892147, 113.266838, 1.025716, 1.013424, 0.557468, 0.063637, 1],[1.878869,-0.988801,0.028579, 243.780426, 105.846382, 1.878869, 1.896841, -0.997999, 0.029114, 1],[1.901071,-1.002995,0.065552, 244.070755, 110.085938, 1.901071, 1.910481, -1.006562, 0.065708, 1],[1.381389,0.369560,0.083893, 69.143921, 115.860832, 1.381389, 1.383224, 0.359882, 0.121258, 1],[1.307210,0.318143,0.061654, 74.457321, 112.876129, 1.307210, 1.332985, 0.324425, 0.062924, 1],[1.278978,0.296857,0.087915, 76.936981, 117.622406, 1.278978, 1.276663, 0.295777, 0.087179, 1],[1.321111,0.240242,0.087150, 87.993340, 117.012825, 1.321111, 1.326163, 0.241225, 0.087403, 1],[1.658762,-0.173830,0.124454, 151.054886, 119.006271, 1.658762, 1.628773, -0.137139, 0.072749, 1],[1.504240,-0.173036,0.098211, 153.307068, 116.863701, 1.504240, 1.511551, -0.173841, 0.098668, 1],[1.642276,-0.234691,0.073184, 159.439255, 112.303825, 1.642276, 1.853186, -0.992318, 0.116026, 0],[1.033430,0.581802,0.095536, 4.144067, 122.837921, 1.033430, 1.047610, 0.588968, 0.097054, 1],[1.060329,0.572685,0.084884, 9.177825, 120.111877, 1.060329, 1.074047, 0.575215, 0.082139, 1],[2.978261,-0.679361,0.318236, 178.183456, 126.007668, 2.978261, 1.645444, -0.139369, 0.071714, 0],[1.840802,-0.970041,0.094128, 243.932648, 113.749557, 1.840802, 1.845971, -0.974761, 0.126676, 1],[1.844403,-0.974725,0.126424, 244.264999, 117.579865, 1.844403, 1.845971, -0.974761, 0.126676, 1],[1.638928,-0.563043,0.158855, 203.579483, 123.823753, 1.638928, 1.617703, -0.555280, 0.157336, 1],[1.467311,-0.020613,0.134181, 131.090530, 122.618332, 1.467311, 1.488139, -0.021068, 0.135616, 1],[1.507221,-0.060426,0.137621, 136.819992, 122.587784, 1.507221, 1.500178, -0.059965, 0.137612, 1],[1.335737,0.260249,0.077265, 85.136200, 115.225761, 1.335737, 1.331833, 0.259380, 0.077057, 1],[1.380656,0.254565,0.064945, 87.436539, 112.848671, 1.380656, 1.377236, 0.253511, 0.063334, 1],[1.598221,-0.508177,0.150931, 197.952164, 123.276054, 1.598221, 1.587440, -0.506247, 0.146156, 1],[1.661426,-0.466963,0.154563, 189.833557, 122.966721, 1.661426, 1.675188, -0.483435, 0.114719, 1],[1.590771,-0.462560,0.173910, 191.970963, 126.551369, 1.590771, 1.606000, -0.451554, 0.148116, 1],[1.346943,0.350631,0.118328, 70.730431, 121.826859, 1.346943, 1.383224, 0.359882, 0.121258, 1],[1.282580,0.291857,0.112547, 77.938049, 121.805038, 1.282580, 1.266327, 0.288001, 0.111247, 1],[1.677763,-0.211828,0.079236, 155.776398, 112.889938, 1.677763, 1.399311, 0.366397, 0.277107, 0],[1.340133,0.270843,0.129936, 83.537712, 123.830681, 1.340133, 1.344395, 0.271694, 0.130120, 1],[1.500945,0.402581,0.064434, 68.991928, 111.944305, 1.500945, 1.449144, 0.388708, 0.063722, 1],[1.827053,-0.958104,0.155575, 243.367691, 121.233162, 1.827053, 1.845971, -0.974761, 0.126676, 1],[1.596169,-0.135509,0.070008, 146.677170, 112.149139, 1.596169, 1.628773, -0.137139, 0.072749, 1],[1.280302,0.283241,0.143994, 79.329369, 127.243210, 1.280302, 1.272407, 0.281448, 0.142834, 1],[1.368310,0.348641,0.177113, 71.944649, 130.976624, 1.368310, 1.334668, 0.340157, 0.173201, 1],[1.336456,0.234297,0.179884, 89.431366, 132.111572, 1.336456, 1.319687, 0.231359, 0.178152, 1],[3.209522,-0.800726,0.284470, 182.886597, 121.999298, 3.209522, 1.286583, 0.292737, 0.112876, 0],[1.450998,0.281562,0.125032, 85.309647, 121.457306, 1.450998, 1.383767, 0.268266, 0.119489, 1],[1.695952,-0.465969,0.157221, 188.445816, 122.894836, 1.695952, 1.708116, -0.452254, 0.162403, 1],[1.326966,0.244411,0.137740, 87.478600, 125.336121, 1.326966, 1.318177, 0.242570, 0.135302, 1],[1.691978,-0.184308,0.218601, 151.964722, 130.923691, 1.691978, 1.657397, -0.178633, 0.212842, 1],[1.646711,-0.613228,0.172629, 209.927048, 125.563156, 1.646711, 1.653009, -0.615573, 0.173630, 1],[1.651309,-0.719253,0.133691, 223.824310, 120.311394, 1.651309, 1.635061, -0.712407, 0.132641, 1],[1.258902,0.301849,0.156537, 75.250175, 129.855713, 1.258902, 1.272407, 0.282968, 0.151478, 1],[1.606219,-0.343880,0.229337, 175.100372, 133.911713, 1.606219, 1.630893, -0.348856, 0.232908, 1],[1.528130,-0.203530,0.197452, 157.301605, 130.926468, 1.528130, 1.560619, -0.207641, 0.201162, 1],[2.007777,-1.105981,0.369354, 249.186676, 142.971603, 2.007777, 1.680993, -0.743934, 0.102586, 0],[1.517741,-0.006727,0.193932, 128.975082, 130.610916, 1.517741, 1.514532, -0.006754, 0.194746, 1],[1.486363,-0.021611,0.203246, 131.198624, 132.582947, 1.486363, 1.461445, -0.021026, 0.199787, 1],[1.493785,-0.039834,0.170226, 133.866562, 127.570282, 1.493785, 1.489875, -0.039911, 0.169471, 1],[1.512447,-0.056721,0.207531, 136.250595, 132.687424, 1.512447, 1.488294, -0.039006, 0.220348, 1],[1.433857,0.039286,0.200396, 121.972328, 133.247223, 1.433857, 1.434251, 0.039312, 0.200669, 1],[2.056807,-1.166165,0.367304, 252.735229, 141.787506, 2.056807, 1.540611, 0.017627, 0.181639, 0],[1.812476,-0.915063,0.177738, 239.071243, 124.073990, 1.812476, 1.853229, -0.918646, 0.144899, 1],[1.495715,-0.039415,0.222714, 133.797409, 135.258362, 1.495715, 1.488294, -0.039006, 0.220348, 1],[1.428817,0.229221,0.177166, 92.706039, 129.778854, 1.428817, 1.399683, 0.233455, 0.215231, 1],[1.288876,0.275011,0.109857, 81.057938, 121.251648, 1.288876, 1.279398, 0.272995, 0.108916, 1],[1.323275,0.055670,0.181077, 118.744667, 132.604782, 1.323275, 1.315378, 0.055459, 0.180327, 1],[1.288837,0.275831,0.139081, 80.916603, 126.240585, 1.288837, 1.287295, 0.275529, 0.138900, 1],[1.682026,-0.603601,0.200535, 206.947739, 128.728897, 1.682026, 1.661610, -0.577533, 0.192943, 1],[1.679124,-0.647705,0.193832, 212.862717, 127.896027, 1.679124, 1.654124, -0.658373, 0.175274, 1],[1.578556,-0.063375,0.123791, 136.832397, 119.752457, 1.578556, 1.575083, -0.084508, 0.167489, 1],[1.330246,0.230493,0.154247, 89.880417, 128.009857, 1.330246, 1.322273, 0.229056, 0.153554, 1],[1.287170,0.269738,0.225912, 81.897087, 141.112350, 1.287170, 1.273022, 0.267062, 0.223294, 1],[1.262526,0.275011,0.155989, 80.078262, 129.681656, 1.262526, 1.272407, 0.276064, 0.156419, 1],[1.982172,-1.137203,0.383141, 254.217392, 145.024551, 1.982172, 1.490123, -0.021533, 0.203931, 0],[1.571639,0.407400,0.285279, 70.971695, 142.433777, 1.571639, 1.525696, 0.423006, 0.258145, 1],[1.272986,0.300198,0.193909, 76.119156, 136.011810, 1.272986, 1.279687, 0.292386, 0.210777, 1],[1.551098,0.001467,0.183080, 127.791954, 128.467209, 1.551098, 1.552547, 0.000904, 0.184052, 1],[1.495926,-0.015951,0.227483, 130.345825, 135.955002, 1.495926, 1.479733, -0.015994, 0.225245, 1],[1.976339,-1.052038,0.505168, 245.109665, 158.733719, 1.976339, 1.980431, -1.050983, 0.506932, 1],[1.923473,-1.056042,0.470346, 248.786255, 156.296494, 1.923473, 1.929523, -1.059622, 0.472301, 1],[1.311670,0.324834,0.290797, 73.517136, 151.273941, 1.311670, 1.322195, 0.322158, 0.298162, 1],[1.309043,0.228059,0.227966, 89.672005, 140.812393, 1.309043, 1.317731, 0.229412, 0.229483, 1],[1.371104,0.265481,0.212348, 85.402367, 136.572144, 1.371104, 1.334048, 0.263288, 0.202108, 1],[1.386104,0.231266,0.212933, 91.293793, 136.296371, 1.386104, 1.376863, 0.232136, 0.213845, 1],[1.443132,0.079446,0.259230, 115.888779, 142.018631, 1.443132, 1.450188, 0.058869, 0.273057, 1],[1.313184,0.262500,0.158463, 84.022873, 129.047607, 1.313184, 1.304986, 0.260966, 0.157785, 1],[1.316114,0.244280,0.157885, 87.166405, 128.891815, 1.316114, 1.313200, 0.243678, 0.157553, 1],[1.264830,0.327758,0.297898, 70.990929, 154.315231, 1.264830, 1.284944, 0.333392, 0.305534, 1],[1.843875,-0.922118,0.515153, 238.021561, 163.964890, 1.843875, 1.880223, -0.940454, 0.524944, 1],[1.297822,0.260104,0.255084, 83.908607, 145.740540, 1.297822, 1.309569, 0.263418, 0.257074, 1],[1.082906,0.396066,0.257432, 47.536327, 154.799072, 1.082906, 1.081112, 0.395568, 0.256969, 1],[1.102479,0.380063,0.257980, 52.158352, 153.980026, 1.102479, 1.092128, 0.376529, 0.255556, 1],[1.349286,0.546228,0.328795, 38.937965, 156.109787, 1.349286, 1.366394, 0.553801, 0.333951, 1],[1.376621,0.534173,0.315109, 42.632927, 152.858047, 1.376621, 1.366394, 0.553801, 0.333951, 1],[1.704488,-0.532841,0.484040, 196.774399, 164.975571, 1.704488, 1.710190, -0.534161, 0.486247, 1],[1.460296,0.059305,0.274925, 119.065407, 143.918716, 1.460296, 1.450188, 0.058869, 0.273057, 1],[1.854893,-0.902826,0.535882, 235.079895, 166.058441, 1.854893, 1.880223, -0.940454, 0.524944, 1],[1.132435,0.442077,0.274787, 42.116924, 155.883392, 1.132435, 1.117565, 0.391127, 0.276727, 1],[1.280250,0.479277,0.287060, 45.640381, 151.828796, 1.280250, 1.266640, 0.474792, 0.283893, 1],[1.338310,0.492198,0.338021, 47.089272, 158.065994, 1.338310, 1.264049, 0.465406, 0.319016, 1],[1.161159,0.406252,0.287633, 51.029068, 156.996552, 1.161159, 1.117565, 0.391127, 0.276727, 1],[1.714698,-0.534228,0.463137, 196.542847, 161.921661, 1.714698, 1.705004, -0.528890, 0.460999, 1],[1.284218,0.292713,0.211719, 77.855186, 138.769638, 1.284218, 1.279687, 0.292386, 0.210777, 1],[1.839935,-0.909075,0.692629, 236.697556, 185.317245, 1.839935, 1.867139, -0.912196, 0.707576, 1],[1.514536,-0.225414,0.580016, 160.743362, 186.752502, 1.514536, 1.533456, -0.228269, 0.586302, 1],[1.645097,-0.338857,0.615642, 173.315643, 184.830261, 1.645097, 1.653211, -0.341012, 0.619713, 1],[1.663220,-0.385457,0.625285, 178.985718, 185.208649, 1.663220, 1.677645, -0.412501, 0.641128, 1],[1.878902,-0.922155,0.628445, 235.974869, 176.084427, 1.878902, 1.832021, -0.923132, 0.593124, 1],[1.805321,-0.912846,0.650493, 239.241241, 181.770325, 1.805321, 1.832021, -0.917916, 0.592750, 1],[1.338415,0.334371,0.434457, 73.038315, 173.913284, 1.338415, 1.340610, 0.334768, 0.435404, 1],[1.520114,-0.191458,0.591682, 155.709000, 188.131805, 1.520114, 1.509782, -0.190087, 0.587697, 1],[1.484489,-0.038703,0.536675, 133.735809, 182.034744, 1.484489, 1.482419, -0.039095, 0.536015, 1],[1.703583,-0.494668,0.678520, 191.881226, 190.123825, 1.703583, 1.710244, -0.500876, 0.683098, 1],[1.566096,-0.228029,0.506642, 160.032776, 173.671417, 1.566096, 1.564012, -0.227866, 0.506042, 1],[1.631040,-0.428055,0.735994, 185.737534, 201.773346, 1.631040, 1.649179, -0.433957, 0.744359, 1],[1.593931,-0.283341,0.496877, 167.107727, 171.080719, 1.593931, 1.596428, -0.284119, 0.497414, 1],[1.501522,-0.080270,0.528498, 139.760956, 179.934479, 1.501522, 1.503923, -0.080216, 0.529294, 1],[1.689014,-0.434062,0.670745, 184.538055, 189.866943, 1.689014, 1.677645, -0.412501, 0.641128, 1],[1.699321,-0.426137,0.651478, 183.169159, 186.842560, 1.699321, 1.677645, -0.418466, 0.637074, 1],[1.500817,-0.101166,0.542017, 142.829620, 181.952530, 1.500817, 1.510668, -0.101576, 0.545682, 1],[1.721079,-0.508149,0.678769, 192.954987, 189.264877, 1.721079, 1.712866, -0.507078, 0.677010, 1],[1.711902,-0.487271,0.652627, 190.620209, 186.370407, 1.711902, 1.709298, -0.490291, 0.654285, 1],[1.520689,-0.119219,0.535287, 145.247498, 179.940582, 1.520689, 1.510668, -0.101576, 0.545682, 1],[1.742492,-0.561749,0.643340, 198.924103, 183.725525, 1.742492, 1.728029, -0.556911, 0.641290, 1],[1.209934,-0.072160,0.555237, 141.120697, 203.457718, 1.209934, 1.224174, -0.074309, 0.561686, 1],[1.699280,-0.544679,0.751938, 198.517792, 199.850922, 1.699280, 1.689183, -0.540227, 0.749016, 1],[1.248359,-0.069435,0.540890, 140.236649, 197.821701, 1.248359, 1.253641, -0.063110, 0.538509, 1],[1.242603,-0.113293,0.529741, 148.058289, 196.289490, 1.242603, 1.240093, -0.097026, 0.536353, 1],[1.410784,-0.098979,0.542040, 143.435013, 187.026657, 1.410784, 1.414644, -0.094957, 0.550316, 1],[1.442593,0.070268,0.568955, 117.283867, 189.267426, 1.442593, 1.437352, 0.069987, 0.567143, 1],[1.437391,0.064966,0.545271, 118.056641, 185.956451, 1.437391, 1.446458, 0.065378, 0.548476, 1],[1.624020,-0.339320,0.711574, 173.966415, 198.894287, 1.624020, 1.608540, -0.308841, 0.705395, 1],[1.547695,-0.209936,0.579985, 157.841782, 184.943039, 1.547695, 1.552973, -0.210652, 0.581949, 1],[1.660030,-0.398849,0.568848, 180.858597, 177.888092, 1.660030, 1.682782, -0.402671, 0.576782, 1],[1.425938,-0.058958,0.594776, 137.096283, 194.264572, 1.425938, 1.420088, -0.057823, 0.594698, 1],[1.698550,-0.501862,0.698981, 193.002365, 193.033661, 1.698550, 1.710244, -0.503897, 0.701854, 1],[1.456032,0.045964,0.546136, 121.055054, 185.018768, 1.456032, 1.443197, 0.045687, 0.541439, 1],[1.236503,-0.093937,0.535178, 144.713440, 197.719437, 1.236503, 1.233378, -0.094154, 0.534466, 1],[1.697979,-0.554383,0.729842, 199.829117, 197.062546, 1.697979, 1.690648, -0.552241, 0.726480, 1],[1.475570,-0.028743,0.554000, 132.285416, 185.098618, 1.475570, 1.482419, -0.039095, 0.536015, 1],[1.692451,-0.397828,0.664163, 179.713226, 188.833893, 1.692451, 1.680950, -0.393058, 0.657634, 1],[1.518540,-0.146840,0.568599, 149.273651, 184.876373, 1.518540, 1.534942, -0.148027, 0.574482, 1],[1.693871,-0.476553,0.666778, 189.894699, 189.101181, 1.693871, 1.713736, -0.475015, 0.673667, 1],[1.712884,-0.490839,0.655435, 191.042557, 186.682999, 1.712884, 1.709298, -0.490291, 0.654285, 1],[1.704814,-0.424980,0.647979, 182.842163, 186.119263, 1.704814, 1.677645, -0.418466, 0.637074, 1],[1.462582,-0.059277,0.612862, 136.916367, 194.686050, 1.462582, 1.420088, -0.057823, 0.594698, 1],[1.277396,-0.164969,0.526167, 156.411774, 193.119385, 1.277396, 1.265297, -0.185617, 0.526042, 1],[1.280807,-0.196019,0.522619, 161.669632, 192.268616, 1.280807, 1.282358, -0.197045, 0.522818, 1],[1.257687,-0.064381,0.540407, 139.261826, 197.030289, 1.257687, 1.253641, -0.063110, 0.538509, 1],[1.336848,0.328757,0.494984, 73.897636, 183.957703, 1.336848, 1.335741, 0.315332, 0.512425, 1],[1.614537,-0.306852,0.655879, 169.812225, 191.871338, 1.614537, 1.636294, -0.309328, 0.665109, 1],[1.363299,-0.084927,0.566111, 141.704956, 193.855133, 1.363299, 1.345016, -0.084091, 0.558291, 1],[1.334426,0.315042,0.511702, 76.060623, 186.861633, 1.334426, 1.335741, 0.315332, 0.512425, 1],[1.332640,0.309735,0.567473, 76.867104, 196.181656, 1.332640, 1.325861, 0.307458, 0.565459, 1],[1.614368,-0.309207,0.707691, 170.137527, 198.941437, 1.614368, 1.608540, -0.308841, 0.705395, 1],[1.711489,-0.575283,0.720448, 201.948593, 195.108597, 1.711489, 1.736926, -0.584824, 0.726531, 1],])
print(cloud[0])


fig = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax31 = fig3.add_subplot(111, projection='3d')
# ax1.set_ylim(-0.2, 0.2)
# ax1.set_xlim(0, 0.4)
# ax1.set_zlim(-0.2, 0.2)
ax21 = fig2.add_subplot(111)
# ax22 = fig2.add_subplot(122)


minval = 99999.0
maxval = 0
for i in range(len(cloud)):
    if (cloud[i][0] < minval):
        minval = cloud[i][0]
    if (cloud[i][0] > maxval):
        maxval = cloud[i][0]
    if (cloud[i][1] < -0.55):
        print(cloud[i])    
print(minval)
print(maxval)



for i in range(len(cloud)):
    if (True):
        color = "blue"
        ax1.scatter(0, 0, 0, color='#A41F22')
        ax1.scatter(cloud[i][0], cloud[i][1], -cloud[i][2], color='#0465A9', s=1)
        ax31.scatter(cloud[i][0], cloud[i][1], -cloud[i][2], color="blue", s=1)
        ax31.scatter(cloud[i][6], cloud[i][7], -cloud[i][8], color="red", s=1)        
        ax21.scatter(-cloud[i][1], -cloud[i][2], color=color, s=2)
        # ax22.scatter(cloud[i][1], cloud[i][0], color=color, s=1)
        if (cloud[i][9] == 1):
            
            ax21.scatter(-cloud[i][7], -cloud[i][8], color="red", s=2)
            ax21.plot((-cloud[i][1],-cloud[i][7]),(-cloud[i][2],-cloud[i][8]))
ax1.plot([1.259,2.03],[0.715,-1.0778],zs=[0.418,0.4451], color='#A41F22')
# ax21.plot([0.715,-1.0778],[-0.418,-0.4451]) 
# ax22.plot([0.715,-1.0778,],[1.259,2.03])     
ax1.set_xlabel("A")
ax1.set_ylabel("B")
ax1.set_zlabel("C")
plt.show()
