import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d as a3
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import math

cloud = np.array([[1.410448,0.456023,0.200111, 56.870110, 133.713043, 1.410448, 1.391170, -0.116283, 0.331773, 0],[1.373537,0.320034,0.000551, 76.740013, 102.588257, 1.373537, 1.615386, 0.166228, 0.164497, 0],[1.700512,-0.138588,-0.731439, 145.929565, 7.871644, 1.700512, 1.667275, -0.247821, -0.729894, 1],[1.767136,-0.428342,-0.546568, 181.326477, 34.454845, 1.767136, 1.486889, 0.124662, 0.461681, 0],[1.346296,0.445372,0.468293, 55.221241, 179.024338, 1.346296, 1.330646, 0.344996, 0.459626, 1],[1.373864,0.294475,0.368956, 80.845116, 161.581711, 1.373864, 1.288912, 0.477380, 0.380998, 0],[1.760960,-0.231799,-0.700372, 156.959076, 15.001231, 1.760960, 1.498511, 0.418129, -0.373911, 0],[1.460966,0.194688,0.175487, 98.682854, 128.925751, 1.460966, 1.594038, -0.439008, -0.711343, 0],[1.677109,-0.384714,0.587679, 178.466095, 179.590561, 1.677109, 1.661456, -0.484436, 0.576379, 1],[1.733932,-0.505702,0.344160, 192.163177, 146.166763, 1.733932, 1.396177, 0.035333, 0.360954, 0],[1.613219,-0.845129,0.504380, 243.253052, 171.284027, 1.613219, 1.484705, -0.545518, 0.350043, 0],[1.798851,-0.665032,0.587172, 209.333649, 174.311310, 1.798851, 1.464144, -0.546192, 0.349941, 0],[1.274368,-0.140366,0.543007, 152.232010, 196.241760, 1.274368, 1.484684, 0.417611, -0.361808, 0],[1.436359,0.056414,0.608266, 119.359322, 195.665161, 1.436359, 1.506319, -0.719588, 0.523167, 0],[1.430513,-0.054136,0.555875, 136.325562, 187.988510, 1.430513, 1.230110, -0.357451, 0.523059, 0],[1.657501,-0.065235,-0.631259, 136.658661, 18.713034, 1.657501, 1.484684, 0.417611, -0.361808, 0],[1.619707,-0.798163,0.482011, 236.412094, 167.970139, 1.619707, 1.162128, 0.395542, 0.038615, 0],[1.427566,0.436376,0.240417, 60.750809, 139.550232, 1.427566, 1.144050, 0.477827, 0.131898, 0],[1.353957,0.359536,-0.013610, 69.580284, 100.288574, 1.353957, 1.436480, -0.390757, 0.479717, 0],[1.509197,0.209191,0.238567, 97.505646, 137.276535, 1.509197, 1.388212, -0.059200, -0.038333, 0],[1.735452,-0.784606,0.047455, 227.463013, 108.515724, 1.735452, 1.416246, -0.397666, 0.199649, 0],[1.476657,0.044597,0.521208, 121.355759, 180.152267, 1.476657, 1.470886, -0.064751, 0.531238, 1],[1.494187,0.238385,0.100725, 92.900856, 117.330437, 1.494187, 1.387914, 0.100667, -0.031055, 0],[1.567387,-0.079318,0.207559, 139.133194, 131.633133, 1.567387, 1.454313, -0.259658, 0.149969, 0],[1.632905,-0.238413,0.547078, 160.121170, 176.207443, 1.632905, 1.601190, -0.349797, 0.539335, 1],[1.419790,0.426928,0.261162, 61.846462, 142.967758, 1.419790, 1.384368, -0.448504, 0.528098, 0],[1.863808,-0.870314,0.142208, 230.730087, 119.285995, 1.863808, 1.382998, -0.075825, 0.313621, 0],[1.709025,-0.444413,0.538535, 185.208618, 171.824783, 1.709025, 1.679665, -0.558196, 0.512207, 1],[1.352843,0.343872,0.332509, 72.079376, 156.572830, 1.352843, 1.364665, -0.005195, 0.339686, 0],[1.341600,0.407909,0.357147, 61.109768, 161.066177, 1.341600, 1.385270, 0.022995, 0.316768, 0],[1.806181,-0.300684,-0.652071, 164.624512, 23.075121, 1.806181, 1.479648, -0.195099, 0.073147, 0],[1.472338,0.014676,0.231529, 125.807060, 137.095520, 1.472338, 1.375258, 0.101318, 0.299824, 0],[1.994003,-1.028270,0.092923, 241.449905, 112.752220, 1.994003, 1.304302, -0.063990, 0.063673, 0],[1.222800,0.290234,0.140441, 75.782532, 127.767509, 1.222800, 1.147006, 0.289182, 0.049016, 0],[1.700196,-0.256947,0.197776, 161.248169, 128.091568, 1.700196, 1.494730, -0.660558, 0.494545, 0],[1.620636,-0.618207,0.150884, 211.921127, 122.982437, 1.620636, 1.389058, -0.053504, -0.035662, 0],[1.702490,-0.133534,-0.730388, 145.255646, 8.117408, 1.702490, 1.669020, -0.232263, -0.721186, 1],[1.554607,-0.144302,0.539726, 148.420853, 178.879257, 1.554607, 1.546078, -0.252367, 0.548016, 1],[1.626369,-0.269506,0.452638, 164.456268, 163.728668, 1.626369, 1.658161, -0.565144, -0.639036, 0],[1.694005,-0.456399,0.505799, 187.272385, 168.187958, 1.694005, 1.678807, -0.567514, 0.498207, 1],[1.331254,0.412266,0.374674, 59.869854, 164.417725, 1.331254, 1.558681, -0.809415, 0.469769, 0],[1.419790,0.426928,0.261162, 61.846462, 142.967758, 1.419790, 1.494730, -0.660558, 0.494545, 0],[1.353957,0.359536,-0.013610, 69.580284, 100.288574, 1.353957, 1.532753, -0.576134, 0.443870, 0],[1.430513,-0.054136,0.555875, 136.325562, 187.988510, 1.430513, 1.230110, -0.357451, 0.523059, 0],[1.554607,-0.144302,0.539726, 148.420853, 178.879257, 1.554607, 1.546078, -0.252367, 0.548016, 1],[1.702490,-0.133534,-0.730388, 145.255646, 8.117408, 1.702490, 1.669020, -0.232263, -0.721186, 1],[1.632905,-0.238413,0.547078, 160.121170, 176.207443, 1.632905, 1.601190, -0.349797, 0.539335, 1],[1.298060,0.597133,-0.428710, 26.795630, 29.840565, 1.298060, 1.591297, -0.357727, -0.690350, 0],[2.058403,-0.941556,-0.800083, 228.632568, 16.987928, 2.058403, 1.625315, -0.744392, -0.679432, 0],[1.773832,-0.834022,0.014526, 231.439865, 104.301598, 1.773832, 1.726044, -0.937800, 0.006629, 1],[1.811397,-0.219256,-0.812719, 154.629333, 3.792678, 1.811397, 1.789193, -0.329111, -0.822721, 1],[1.745550,-0.266064,-0.696943, 161.533356, 14.660965, 1.745550, 1.471370, 0.073633, -0.429969, 0],[1.702490,-0.133479,-0.729637, 145.248428, 8.214546, 1.702490, 1.669020, -0.232263, -0.721186, 1],[1.313204,0.729725,-0.418751, 5.749823, 32.347008, 1.313204, 1.292225, 0.621230, -0.409477, 1],[1.320495,0.637897,-0.409166, 21.723680, 34.331196, 1.320495, 1.414029, -0.395105, 0.112992, 0],[1.672720,-0.209929,-0.585161, 155.610397, 25.538210, 1.672720, 1.384345, -0.025114, -0.055741, 0],[1.607259,-0.469469,0.143397, 192.260483, 122.127975, 1.607259, 1.596772, -0.574026, 0.139247, 1],[1.679651,-0.573495,0.118929, 203.116180, 118.077301, 1.679651, 1.658350, -0.682828, 0.105562, 1],[1.698907,-0.624366,0.180722, 208.852325, 125.902603, 1.698907, 1.557018, -0.845821, 0.099388, 0],[1.784596,-0.943590,0.433097, 244.323181, 155.890991, 1.784596, 1.335144, -0.108315, 0.123712, 0],[1.958161,-0.971875,0.480767, 237.190521, 156.514297, 1.958161, 1.169997, 0.296948, 0.050520, 0],[1.797995,-0.658101,-0.047439, 208.524246, 96.695480, 1.797995, 1.404552, -0.552353, 0.063131, 0],[1.672651,-0.042837,-0.621834, 133.634201, 20.711597, 1.672651, 1.155077, 0.289900, 0.049889, 0],[1.233296,0.224552,0.176983, 87.943489, 134.070877, 1.233296, 1.506704, -0.833727, 0.428176, 0],[1.738109,-0.536866,0.104384, 195.953461, 115.712387, 1.738109, 1.701533, -0.647918, 0.101556, 1],[1.711042,-0.307366,-0.592146, 167.520050, 26.363842, 1.711042, 1.672714, -0.408524, -0.593133, 1],[1.443342,0.049162,0.187351, 120.506546, 131.056854, 1.443342, 1.289030, 0.023088, 0.124866, 0],[1.831731,-0.850970,0.161294, 230.205673, 121.872253, 1.831731, 1.790939, -0.956237, 0.155757, 1],[1.606334,-0.435283,0.189694, 187.615448, 128.480057, 1.606334, 1.448557, -0.311654, 0.112089, 0],[1.552122,-0.207449,0.218638, 157.404160, 133.490005, 1.552122, 1.387462, 0.066783, 0.185311, 0],[1.477106,0.004947,0.142647, 127.263268, 123.745796, 1.477106, 1.519275, -0.786617, 0.161650, 0],[1.522027,0.453088,0.143568, 62.508774, 123.251884, 1.522027, 1.383029, 0.035769, 0.296110, 0],[1.438362,0.408223,0.030897, 65.561600, 107.225800, 1.438362, 1.257528, 0.538816, 0.394700, 0],[1.506126,-0.054485,0.166639, 135.958572, 126.841034, 1.506126, 1.381226, 0.133131, 0.305002, 0],[1.832573,-0.931359,0.108687, 239.809448, 115.547867, 1.832573, 1.410530, -0.161012, 0.236435, 0],[1.606605,-0.144700,0.099985, 147.814514, 116.191368, 1.606605, 1.296738, 0.077535, 0.072595, 0],[1.410806,0.256375,0.125682, 88.021111, 122.098770, 1.410806, 1.393786, 0.175995, 0.128821, 1],[1.579936,-0.188318,0.102556, 154.222504, 116.780472, 1.579936, 1.413949, -0.076553, 0.238818, 0],[1.422398,0.400883,0.210329, 65.996063, 135.031250, 1.422398, 1.312194, 0.078386, 0.132519, 0],[1.629147,-0.518825,0.185533, 198.062134, 127.554367, 1.629147, 1.421155, -0.198742, 0.161227, 0],[1.240230,0.273766,0.105923, 79.437668, 121.289375, 1.240230, 1.597715, -0.756601, 0.463419, 0],[1.381446,0.385723,-0.036678, 66.572281, 96.658890, 1.381446, 1.191383, -0.422285, 0.512976, 0],[1.332260,0.325784,0.239350, 74.202263, 142.024551, 1.332260, 1.309520, 0.236334, 0.233764, 1],[1.269105,0.284486,0.193699, 78.684166, 136.077835, 1.269105, 1.349138, 0.056793, 0.189286, 0],[1.461066,0.434354,0.370433, 62.597176, 158.277954, 1.461066, 1.419485, -0.413307, 0.514696, 0],[1.325679,0.274197,0.149153, 82.496231, 127.252419, 1.325679, 1.304790, 0.172624, 0.140888, 1],[1.453212,0.423382,0.287914, 63.904747, 146.086990, 1.453212, 1.209467, -0.373026, 0.533636, 0],[1.482375,0.145336,0.190067, 106.430634, 130.708008, 1.482375, 1.400755, -0.305451, 0.531578, 0],[1.392206,0.369545,0.324758, 69.603508, 153.819107, 1.392206, 1.363007, 0.269478, 0.306481, 1],[1.209656,0.084229,0.164064, 112.681366, 132.338287, 1.209656, 1.220346, -0.298459, 0.537033, 0],[1.441209,0.143345,0.286574, 106.118393, 146.245407, 1.441209, 1.242105, 0.485455, 0.444627, 0],[1.233945,-0.020891,0.547378, 131.724609, 200.092010, 1.233945, 1.587635, -0.768122, 0.600973, 0],[1.639903,-0.322988,0.500505, 171.330246, 169.644882, 1.639903, 1.257528, 0.538816, 0.394700, 0],[1.600609,-0.264229,0.636648, 164.317719, 190.005722, 1.600609, 1.494699, -0.540584, 0.485120, 0],[1.712852,-0.433855,0.683723, 183.724655, 190.317902, 1.712852, 1.688096, -0.544980, 0.674084, 1],[1.283971,0.320312,0.086471, 73.116653, 117.316231, 1.283971, 1.284091, 0.224673, 0.081711, 1],[1.701174,-0.472403,0.489009, 189.092361, 165.739914, 1.701174, 1.680816, -0.584103, 0.485502, 1],[1.494619,0.258023,0.078170, 90.020340, 114.006218, 1.494619, 1.498996, -0.597568, 0.608700, 0],[1.885112,-0.850682,0.579328, 227.277985, 170.109863, 1.885112, 1.609535, -0.806842, 0.579123, 0],[1.533205,-0.197332,0.498925, 156.315231, 174.090912, 1.533205, 1.523500, -0.292709, 0.492164, 1],[1.715520,-0.823543,0.484297, 233.612000, 164.606674, 1.715520, 1.240591, -0.539743, 0.514366, 0],[1.706297,-0.491624,0.708775, 191.387100, 193.885269, 1.706297, 1.401782, -0.450033, 0.539616, 0],[1.644551,-0.306294,0.699133, 168.974579, 196.026611, 1.644551, 1.498182, -0.502553, 0.539225, 0],[1.301842,-0.164039,0.532128, 155.721115, 192.425018, 1.301842, 1.286411, -0.246006, 0.524784, 1],[1.706654,-0.428374,0.628626, 183.220520, 183.534485, 1.706654, 1.195913, 0.548373, 0.364258, 0],[1.848170,-0.895469,0.700110, 234.593613, 185.838715, 1.848170, 1.204071, 0.557316, 0.429749, 0],[1.558239,-0.156520,0.559864, 150.098328, 181.544418, 1.558239, 1.544922, -0.260603, 0.559197, 1],[1.456868,0.379908,0.494764, 70.630447, 177.213715, 1.456868, 1.403813, -0.166372, 0.549398, 0],[1.468826,-0.000702,0.558719, 128.105103, 186.184631, 1.468826, 1.458132, -0.106114, 0.552904, 1],[1.395350,0.087533,0.557700, 114.198997, 190.430664, 1.395350, 1.512195, -0.465564, 0.489242, 0],[1.497429,-0.009422,0.526753, 129.384216, 179.889755, 1.497429, 1.260979, -0.387854, 0.515111, 0],[1.460989,0.069620,0.538254, 117.516396, 183.551895, 1.460989, 1.427582, -0.036747, 0.541134, 1],[1.523547,-0.139131,0.591313, 148.090500, 187.885483, 1.523547, 1.510551, -0.519789, 0.610019, 0],[1.664884,-0.336720,0.591175, 172.494659, 180.618607, 1.664884, 1.624650, -0.450178, 0.578299, 1],[1.692608,-0.364430,0.641100, 175.367432, 185.828140, 1.692608, 1.260584, -0.454724, 0.516576, 0],[1.321733,0.331328,0.152524, 72.851074, 127.887276, 1.321733, 1.308420, 0.235577, 0.148450, 1],[1.366768,0.401773,0.336168, 63.329056, 156.610916, 1.366768, 1.260979, -0.387854, 0.515111, 0],[1.344963,0.367305,0.305287, 67.918716, 152.436737, 1.344963, 1.591297, -0.357727, -0.690350, 0],[1.308749,0.233084,0.246763, 88.818634, 143.980774, 1.308749, 1.316255, 0.496200, -0.422470, 0],[1.240230,0.273766,0.105923, 79.437668, 121.289375, 1.240230, 1.349138, 0.056793, 0.189286, 0],[1.606605,-0.144700,0.099985, 147.814514, 116.191368, 1.606605, 1.400235, 0.034212, -0.421257, 0],[1.600609,-0.264229,0.636648, 164.317719, 190.005722, 1.600609, 1.320498, 0.437597, -0.292852, 0],[1.692608,-0.364430,0.641100, 175.367432, 185.828140, 1.692608, 1.511176, -0.434182, 0.546095, 0],[1.797995,-0.658101,-0.047439, 208.524246, 96.695480, 1.797995, 1.579285, -0.767056, 0.673423, 0],[1.233945,-0.020891,0.547378, 131.724609, 200.092010, 1.233945, 1.290499, 0.602299, -0.420608, 0],[1.644551,-0.306294,0.699133, 168.974579, 196.026611, 1.644551, 1.516929, -0.716074, -0.568753, 0],[1.301842,-0.164039,0.532128, 155.721115, 192.425018, 1.301842, 1.286411, -0.246006, 0.524784, 1],[1.209656,0.084229,0.164064, 112.681366, 132.338287, 1.209656, 1.300378, 0.554948, -0.386567, 0],[1.672651,-0.042837,-0.621834, 133.634201, 20.711597, 1.672651, 1.379543, 0.136623, -0.435450, 0],[1.715520,-0.823543,0.484297, 233.612000, 164.606674, 1.715520, 1.525803, -0.340912, -0.440704, 0],[1.523547,-0.139131,0.591313, 148.090500, 187.885483, 1.523547, 1.496250, -0.579142, -0.562581, 0],[1.558239,-0.156520,0.559864, 150.098328, 181.544418, 1.558239, 1.544922, -0.260603, 0.559197, 1],[1.392206,0.369545,0.324758, 69.603508, 153.819107, 1.392206, 1.363007, 0.269478, 0.306481, 1],[1.381446,0.385723,-0.036678, 66.572281, 96.658890, 1.381446, 1.415267, 0.154803, -0.420397, 0],[1.696607,-0.307007,-0.442868, 167.809738, 45.073063, 1.696607, 1.591297, -0.357727, -0.690350, 0],[1.924188,-0.705323,-0.760141, 208.642380, 15.590134, 1.924188, 1.455097, -0.256529, -0.452681, 0],[1.266714,0.701912,-0.411507, 6.093479, 31.030443, 1.266714, 1.328473, 0.460146, -0.365844, 0],[1.304381,0.610973,-0.423658, 24.951929, 31.044947, 1.304381, 1.314888, 0.452512, -0.402600, 0],[1.333684,0.692044,-0.426044, 13.842710, 32.221287, 1.333684, 1.338818, 0.414485, -0.440191, 0],[2.069474,-0.930040,-0.783085, 226.870026, 19.252396, 2.069474, 1.408410, 0.022473, -0.438728, 0],[1.754641,-0.274054,-0.769912, 162.361328, 5.967008, 1.754641, 1.714571, -0.379026, -0.777267, 1],[1.246221,0.705480,-0.422366, 3.458953, 27.938177, 1.246221, 1.432252, 0.145536, -0.418973, 0],[1.278038,0.642151,-0.422026, 17.460840, 29.852890, 1.278038, 1.327199, 0.432186, -0.425410, 0],[1.310514,0.630698,-0.413734, 22.122883, 33.045292, 1.310514, 1.416801, 0.088056, -0.422330, 0],[1.585751,-0.031755,-0.575455, 132.405548, 22.663866, 1.585751, 1.550571, -0.131435, -0.583321, 1],[1.365873,0.407313,-0.406069, 62.394497, 37.094742, 1.365873, 1.594038, -0.439008, -0.711343, 0],[1.690681,-0.122599,-0.741759, 143.953247, 5.978590, 1.690681, 1.659350, -0.220547, -0.742683, 1],[1.657228,-0.231672,-0.551980, 158.754913, 29.223616, 1.657228, 1.284032, 0.402378, -0.000152, 0],[1.465534,0.306037,-0.423103, 82.058990, 38.985523, 1.465534, 1.312194, -0.055711, 0.097374, 0],[1.448093,0.281951,-0.430185, 85.164925, 37.144592, 1.448093, 1.433535, 0.200941, -0.421220, 0],[1.331950,0.580710,-0.414140, 32.083305, 34.095978, 1.331950, 1.141443, 0.303978, 0.030325, 0],[1.404527,0.537108,-0.429361, 43.869366, 35.246506, 1.404527, 1.374311, 0.436626, -0.415302, 1],[1.426226,0.306321,-0.431097, 80.748978, 36.001888, 1.426226, 1.191201, 0.336598, 0.017370, 0],[1.648352,-0.155799,-0.426176, 148.793976, 45.619720, 1.648352, 1.616911, -0.258018, -0.445379, 1],[1.566833,0.001877,-0.588408, 127.736519, 19.881359, 1.566833, 1.495752, -0.625230, 0.031708, 0],[1.781224,-0.209358,-0.812321, 153.857895, 2.169745, 1.781224, 1.650016, -0.645262, -0.468623, 0],[1.705656,-0.345986,-0.706704, 172.626175, 11.347493, 1.705656, 1.519199, -0.290661, -0.577647, 0],[1.815215,-0.225288,-0.798647, 155.304398, 5.705737, 1.815215, 1.789193, -0.329111, -0.822721, 1],[1.711056,-0.170831,-0.755489, 149.964630, 5.362609, 1.711056, 1.373294, -0.021018, 0.021279, 0],[1.644528,-0.313744,-0.551875, 169.971741, 28.671816, 1.644528, 1.362697, -0.055939, 0.023044, 0],[1.774166,-0.282632,-0.735279, 163.046875, 11.324020, 1.774166, 1.375957, -0.050610, 0.022734, 0],[1.813638,-0.559887,-0.449716, 195.916046, 47.948071, 1.813638, 1.359840, -0.148096, 0.143541, 0],[1.611332,-0.131283,-0.435216, 145.924423, 43.078594, 1.611332, 1.138983, 0.293649, 0.069101, 0],[1.648352,-0.156551,-0.428735, 148.894302, 45.278179, 1.648352, 1.616911, -0.258018, -0.445379, 1],[1.785221,-0.331917,-0.696328, 168.903503, 16.688633, 1.785221, 1.072697, 0.347077, 0.036138, 0],[1.657386,-0.084696,-0.695404, 139.242462, 10.192699, 1.657386, 1.641322, -0.185268, -0.697931, 1],[1.709052,-0.135484,-0.679437, 145.440384, 15.038598, 1.709052, 1.444691, -0.319729, 0.089359, 0],[1.316398,0.555175,-0.422335, 35.217594, 31.918201, 1.316398, 1.110725, 0.261736, 0.049190, 0],[1.413802,0.465263,-0.419662, 55.601059, 37.196983, 1.413802, 1.353543, 0.053490, 0.047630, 0],[1.684946,-0.147583,-0.699354, 147.269653, 11.186702, 1.684946, 1.667275, -0.255413, -0.719473, 1],[1.660606,-0.234386,-0.548877, 159.051849, 29.783821, 1.660606, 1.395409, 0.042753, 0.049816, 0],[1.722520,-0.357806,-0.598053, 173.698990, 26.116777, 1.722520, 1.686274, -0.459085, -0.610624, 1],[1.496828,-0.002936,-0.109649, 128.431503, 86.384125, 1.496828, 1.326021, -0.080091, 0.089228, 0],[1.755516,-0.276420,-0.689831, 162.640747, 16.050938, 1.755516, 1.315203, -0.069957, 0.182556, 0],[1.620901,-0.206818,-0.557452, 156.070831, 26.838776, 1.620901, 1.327843, 0.073471, 0.099938, 0],[1.404873,0.262009,-0.060269, 86.970039, 93.062027, 1.404873, 1.383804, 0.037788, 0.032044, 0],[1.410935,0.242916,-0.073621, 90.123314, 91.020630, 1.410935, 1.452474, -0.431843, 0.139525, 0],[1.648184,-0.147380,-0.123815, 147.672379, 85.973145, 1.648184, 1.314942, 0.029339, 0.101981, 0],[1.833622,-0.876166,0.001491, 233.123428, 102.678909, 1.833622, 1.432480, -0.427970, 0.158999, 0],[1.428866,0.407363,-0.047432, 65.279045, 95.196915, 1.428866, 1.394089, 0.296832, -0.045871, 1],[1.418270,0.405541,0.047721, 65.093018, 109.902367, 1.418270, 1.520298, -0.823601, 0.431916, 0],[1.643950,-0.335292,-0.086872, 172.870102, 90.874405, 1.643950, 1.330757, -0.115589, 0.059040, 0],[1.395032,0.268485,0.046940, 85.659309, 109.902557, 1.395032, 1.357180, -0.136802, 0.186549, 0],[1.498452,0.436492,-0.032820, 63.915062, 97.681389, 1.498452, 1.190127, 0.429152, 0.079615, 0],[1.798608,-0.902477,0.086872, 238.388107, 113.125954, 1.798608, 1.254669, 0.439085, 0.076230, 0],[1.774919,-0.885291,0.021767, 237.731171, 105.198036, 1.774919, 1.723683, -0.985710, 0.012336, 1],[1.828097,-0.925042,0.128191, 239.322968, 117.926971, 1.828097, 1.187205, 0.349692, 0.041954, 0],[1.724728,-0.828342,0.118859, 233.660263, 117.661209, 1.724728, 1.676179, -0.942586, 0.115719, 1],[1.660926,-0.798806,0.044232, 233.806839, 108.358803, 1.660926, 1.401226, -0.023081, 0.045111, 0],[1.397099,0.265940,0.136101, 86.122681, 123.931641, 1.397099, 1.393786, 0.175995, 0.128821, 1],[1.619936,-0.248489,-0.099233, 161.746811, 89.023399, 1.619936, 1.357278, 0.035987, 0.072491, 0],[1.645427,-0.187659,0.071172, 153.090744, 112.015945, 1.645427, 1.300571, -0.050092, 0.176412, 0],[1.351428,0.350532,0.092279, 70.936600, 117.522148, 1.351428, 1.350059, 0.255524, 0.094882, 1],[1.606605,-0.143606,0.100464, 147.664612, 116.256981, 1.606605, 1.448557, -0.311654, 0.112089, 0],[1.501414,0.014321,0.031293, 125.901596, 107.085289, 1.501414, 1.263636, 0.598531, 0.367207, 0],[1.522516,-0.007631,0.044436, 129.102722, 108.920952, 1.522516, 1.448557, -0.311654, 0.112089, 0],[1.643005,-0.523191,0.107720, 198.055832, 116.923767, 1.643005, 1.477658, -0.799450, 0.123149, 0],[1.682104,-0.572954,0.113113, 202.935898, 117.293839, 1.682104, 1.658350, -0.682828, 0.105562, 1],[1.553623,-0.176949,0.097695, 153.056747, 116.334091, 1.553623, 1.281278, -0.060294, 0.214123, 0],[1.607259,-0.470203,0.144940, 192.360916, 122.339279, 1.607259, 1.596772, -0.574026, 0.139247, 1],[1.210866,0.233428,0.082870, 85.588875, 117.556557, 1.210866, 1.443985, -0.214690, 0.154817, 0],[1.249847,0.320805,0.113527, 71.531464, 122.483192, 1.249847, 1.291365, -0.059821, 0.206689, 0],[1.199671,0.275301,0.116082, 77.514381, 123.787598, 1.199671, 1.442587, -0.312459, 0.141660, 0],[1.638422,-0.426467,0.150810, 185.264069, 122.750053, 1.638422, 1.436975, -0.273490, 0.156600, 0],[1.662120,-0.569571,0.167945, 203.389053, 124.729401, 1.662120, 1.633773, -0.677536, 0.164747, 1],[1.686713,-0.605137,0.187628, 206.928757, 126.972496, 1.686713, 1.459848, -0.241064, 0.117914, 0],[1.466347,0.419657,0.109660, 65.037720, 118.952545, 1.466347, 1.457581, 0.313535, 0.104483, 1],[1.312921,0.260122,0.077138, 84.412643, 115.425621, 1.312921, 1.289030, 0.023088, 0.124866, 0],[1.662781,-0.128928,0.125851, 145.058319, 119.151184, 1.662781, 1.621146, -0.228907, 0.112318, 1],[1.666301,-0.161376,0.070525, 149.306274, 111.811287, 1.666301, 1.259202, 0.610607, 0.389519, 0],[1.715965,-0.584765,0.073536, 202.971420, 111.927841, 1.715965, 1.434578, -0.497623, 0.506658, 0],[1.437914,0.082005,0.145378, 115.453232, 124.742805, 1.437914, 1.431035, -0.380267, 0.524917, 0],[1.351545,0.367279,0.064234, 68.215561, 112.955795, 1.351545, 1.227550, 0.401446, 0.378817, 0],[1.366027,0.352691,0.152201, 71.198730, 127.012138, 1.366027, 1.357375, 0.257587, 0.139388, 1],[1.240230,0.272815,0.105044, 79.606354, 121.133400, 1.240230, 1.277694, 0.475231, 0.482765, 0],[1.348155,0.300931,0.149264, 78.892273, 126.857849, 1.348155, 1.336205, 0.187625, 0.146938, 1],[3.151577,-0.689319,0.265120, 176.118805, 121.007042, 3.151577, 1.272017, 0.392975, 0.512595, 0],[1.487493,0.041931,0.138979, 121.798424, 123.055031, 1.487493, 1.332669, 0.034770, 0.123889, 0],[1.502540,0.021624,0.167776, 124.833893, 127.065529, 1.502540, 1.490392, -0.081393, 0.158519, 1],[1.512843,-0.023149,0.135265, 131.366409, 122.170509, 1.512843, 1.307623, 0.093610, 0.103708, 0],[1.406104,0.107558,0.194112, 111.171364, 132.870895, 1.406104, 1.383838, 0.009049, 0.182528, 1],[1.292489,0.315735,0.185367, 74.257401, 134.052032, 1.292489, 1.231412, 0.404446, 0.353005, 0],[1.907396,-0.906162,0.205871, 232.517197, 126.245224, 1.907396, 1.308881, 0.385889, 0.366041, 0],[2.347018,-0.482142,0.253029, 173.194016, 126.217964, 2.347018, 1.240591, -0.539743, 0.514366, 0],[1.255202,0.447945,0.277061, 49.488453, 151.060699, 1.255202, 1.490875, -0.643475, 0.450200, 0],[1.424036,0.063059,0.134334, 118.258034, 123.253334, 1.424036, 1.412417, -0.289380, 0.554901, 0],[1.526875,0.021109,0.135806, 124.958450, 122.067566, 1.526875, 1.500796, -0.080385, 0.128357, 1],[1.805656,-0.878414,0.173711, 235.025467, 123.664894, 1.805656, 1.432104, -0.396681, 0.510413, 0],[1.500334,-0.041793,0.154860, 134.128250, 125.207726, 1.500334, 1.465838, -0.627449, 0.465142, 0],[1.581511,-0.172117,0.254956, 151.942764, 137.966293, 1.581511, 1.362282, 0.037294, 0.120536, 0],[1.326143,0.271483,0.145484, 82.962471, 126.634956, 1.326143, 1.304790, 0.172624, 0.140888, 1],[1.333603,0.341160,0.234650, 71.720055, 141.209366, 1.333603, 1.309520, 0.236334, 0.233764, 1],[1.152306,0.210025,0.206145, 87.901649, 141.857422, 1.152306, 1.408118, -0.293806, 0.585048, 0],[1.962441,-0.988692,0.487718, 238.837631, 157.175720, 1.962441, 1.441907, -0.398276, 0.509136, 0],[1.434521,0.041376,0.201615, 121.654510, 133.419952, 1.434521, 1.418723, -0.060051, 0.216668, 1],[1.507436,0.020494,0.205379, 125.009079, 132.473618, 1.507436, 1.473663, -0.080448, 0.191444, 1],[1.581038,-0.128335,0.203929, 145.857697, 130.876480, 1.581038, 1.406381, -0.259260, 0.195223, 0],[2.011585,-1.064032,0.324821, 244.369446, 138.024490, 2.011585, 1.297362, 0.075994, 0.163444, 0],[2.021480,-1.089458,0.352299, 246.566986, 140.841141, 2.021480, 1.349138, 0.056793, 0.189286, 0],[1.837996,-0.771155,0.215655, 220.303864, 128.312973, 1.837996, 1.446274, 0.079336, 0.328186, 0],[1.686265,-0.707540,0.137417, 220.309784, 120.428169, 1.686265, 1.639688, -0.805672, 0.130095, 1],[1.758063,-0.775425,0.140654, 225.034897, 120.101105, 1.758063, 1.271783, 0.490038, 0.559735, 0],[1.510296,-0.019963,0.203167, 130.907944, 132.094696, 1.510296, 1.242710, 0.424483, 0.560089, 0],[1.464436,0.100993,0.171069, 112.828026, 128.199493, 1.464436, 1.287635, 0.409822, 0.432726, 0],[1.352395,0.061715,0.187357, 117.960579, 132.978256, 1.352395, 1.352095, 0.017201, 0.213283, 0],[1.632111,-0.306605,0.471927, 169.328781, 166.113327, 1.632111, 1.621482, -0.401264, 0.469441, 1],[1.162340,0.217984,0.142994, 86.741501, 129.564880, 1.162340, 1.400755, -0.305451, 0.531578, 0],[1.282985,0.249641,0.199799, 85.192802, 136.760574, 1.282985, 1.273072, 0.159458, 0.211987, 1],[1.297249,0.250189,0.220446, 85.570457, 139.885391, 1.297249, 1.273072, 0.159458, 0.211987, 1],[1.372389,0.362546,0.210184, 69.882339, 136.193375, 1.372389, 1.370147, 0.265929, 0.204946, 1],[1.478753,0.179654,0.272918, 101.272156, 143.103043, 1.478753, 1.454303, 0.058198, 0.273433, 1],[1.473064,0.103928,0.225619, 112.478531, 136.195801, 1.473064, 1.282520, 0.429468, 0.511604, 0],[1.418348,0.059022,0.215065, 118.845024, 135.858688, 1.418348, 1.480624, -0.618436, 0.506577, 0],[1.626627,-0.193105,0.245025, 154.117279, 135.639496, 1.626627, 1.263599, 0.515453, 0.341501, 0],[1.776194,-0.644988,0.223009, 207.888412, 130.122025, 1.776194, 1.475457, -0.614732, 0.546466, 0],[1.648285,-0.537444,0.191906, 199.733780, 128.114151, 1.648285, 1.560210, -0.676258, 0.563093, 0],[1.429065,0.186252,0.282698, 99.327103, 146.020477, 1.429065, 1.575029, -0.611545, 0.449131, 0],[1.078932,0.409621,0.258202, 44.476200, 155.148849, 1.078932, 1.561814, -0.691353, 0.435359, 0],[1.215869,0.458045,0.273470, 45.121025, 151.981812, 1.215869, 1.603595, -0.674395, 0.448107, 0],[1.147465,0.382233,0.278177, 54.715683, 155.834106, 1.147465, 1.324258, 0.313384, 0.297952, 0],[1.276283,0.399212,0.304952, 59.185623, 155.066284, 1.276283, 1.563983, -0.840445, 0.444360, 0],[1.308749,0.234377,0.246740, 88.601402, 143.976898, 1.308749, 1.283132, 0.418956, 0.505437, 0],[1.480557,0.046818,0.272388, 121.043137, 142.974915, 1.480557, 1.561814, -0.691353, 0.435359, 0],[1.484098,0.018550,0.225204, 125.250183, 135.883774, 1.484098, 1.615548, -0.779560, 0.454049, 0],[1.097978,0.396935,0.258108, 48.466801, 154.216721, 1.097978, 1.457422, -0.243797, 0.532754, 0],[1.328204,0.316338,0.235927, 75.602730, 141.578323, 1.328204, 1.309520, 0.236334, 0.233764, 1],[1.099489,0.382318,0.256179, 51.500862, 153.759628, 1.099489, 1.511011, -0.423956, 0.532774, 0],[1.262115,0.287020,0.197245, 77.969391, 136.881821, 1.262115, 1.194810, 0.500507, 0.456515, 0],[1.131969,0.371604,0.259332, 55.778141, 152.901672, 1.131969, 1.381226, 0.133131, 0.305002, 0],[1.326666,0.341137,0.251050, 71.429535, 144.131363, 1.326666, 1.309520, 0.236334, 0.233764, 1],[1.292924,0.373969,0.305209, 64.366600, 154.433395, 1.292924, 2.193965, -1.064243, 0.384854, 0],[1.234182,0.359064,0.293922, 63.994740, 154.893356, 1.234182, 1.227284, 0.268069, 0.282807, 1],[1.304785,0.359963,0.290828, 67.306580, 151.536621, 1.304785, 1.512702, -0.440667, 0.552832, 0],[1.938650,-0.924676,0.503933, 232.933136, 159.686813, 1.938650, 1.605693, -0.775297, 0.534194, 0],[1.896356,-0.863865,0.552891, 228.218689, 166.641937, 1.896356, 1.545060, -0.465854, 0.488989, 0],[1.979915,-0.936597,0.545958, 232.070801, 163.164612, 1.979915, 1.695129, -0.461571, 0.195371, 0],[1.479047,0.012362,0.534956, 126.161186, 182.071777, 1.479047, 1.460936, -0.094198, 0.545451, 1],[1.519013,-0.062397,0.546494, 137.037003, 181.649124, 1.519013, 1.512677, 0.211100, 0.149547, 0],[1.562988,-0.116091,0.563428, 144.340561, 181.805817, 1.562988, 1.526577, -0.220966, 0.566551, 1],[1.569400,-0.136100,0.574742, 147.078598, 183.067917, 1.569400, 1.471094, -0.526241, 0.564434, 0],[1.458014,0.396468,0.422035, 68.176834, 166.180862, 1.458014, 1.456459, 0.291235, 0.407063, 1],[1.530103,-0.189376,0.491115, 155.228699, 173.113159, 1.530103, 1.523500, -0.292709, 0.492164, 1],[1.718374,-0.484775,0.480483, 190.064819, 164.015274, 1.718374, 1.680816, -0.584103, 0.485502, 1],[1.495406,-0.012437,0.533560, 129.829697, 180.995865, 1.495406, 1.495215, -0.358790, 0.520780, 0],[1.497609,0.422942,0.419606, 65.869392, 164.140533, 1.497609, 1.461601, -0.107527, 0.541816, 0],[1.457114,0.403519,0.551326, 67.075378, 185.741119, 1.457114, 1.195913, 0.548373, 0.364258, 0],[1.426167,0.400537,0.184600, 66.213371, 130.976334, 1.426167, 1.696124, 0.100477, 0.084294, 0],[1.380188,0.364382,0.177550, 69.917976, 130.801193, 1.380188, 1.354416, 0.271709, 0.180467, 1],[1.198901,0.268532,0.151497, 78.724068, 130.299896, 1.198901, 1.257556, 0.476424, 0.543539, 0],[1.626818,-0.382935,0.727858, 179.785568, 200.930634, 1.626818, 1.609718, -0.494357, 0.719502, 1],[1.415682,0.117682,0.592067, 109.711967, 194.508545, 1.415682, 1.377217, 0.167234, 0.217382, 0],[1.687261,-0.454265,0.756022, 187.231140, 201.076874, 1.687261, 1.475773, -0.414322, 0.568453, 0],[1.413042,0.116684,0.561991, 109.833214, 189.997757, 1.413042, 1.405532, 0.004911, 0.544618, 1],[1.538478,-0.091078,0.564104, 141.024063, 183.166016, 1.538478, 1.514288, -0.185877, 0.547862, 1],[1.740887,-0.533207,0.513747, 195.382660, 167.423370, 1.740887, 1.224651, -0.349295, 0.523502, 0],[1.429655,0.368519,0.599300, 71.291115, 194.722336, 1.429655, 1.235018, 0.567134, 0.513973, 0],[1.539090,-0.159799,0.598051, 150.841919, 187.986389, 1.539090, 1.493915, -0.186804, 0.538492, 0],[1.307677,-0.167523,0.528023, 156.183563, 191.333115, 1.307677, 1.483824, -0.434460, 0.507287, 0],[1.644948,-0.276779,0.662444, 165.017166, 191.097092, 1.644948, 1.475063, -0.430008, 0.533815, 0],[1.688038,-0.357993,0.626033, 174.656754, 184.090103, 1.688038, 1.250556, 0.505213, 0.504215, 0],[1.664637,-0.358498,0.663891, 175.379456, 190.240494, 1.664637, 1.536570, -0.680856, 0.693847, 0],[1.650495,-0.316055,0.709162, 170.128067, 197.026627, 1.650495, 1.233728, -0.409694, 0.509175, 0],[1.566337,-0.170182,0.580228, 151.902863, 183.995972, 1.566337, 1.540290, -0.270057, 0.570199, 1],[1.699011,-0.312354,0.630834, 168.445862, 184.184860, 1.699011, 1.235582, -0.284894, 0.528781, 0],[1.270968,-0.171435,0.564109, 157.674789, 200.145309, 1.270968, 1.362035, -0.265871, 0.559397, 0],[1.703256,-0.441505,0.639074, 185.026688, 185.045593, 1.703256, 1.229480, -0.280945, 0.526186, 0],[1.718509,-0.452977,0.669844, 185.989151, 188.252029, 1.718509, 1.688096, -0.544980, 0.674084, 1],[1.623658,-0.300451,0.489646, 168.710114, 168.845322, 1.623658, 1.649885, -0.545021, 0.475468, 0],[1.709745,-0.497333,0.577843, 191.993927, 176.853500, 1.709745, 1.691333, -0.612039, 0.574811, 1],[1.414363,0.096111,0.554894, 113.050140, 188.812088, 1.414363, 1.403844, -0.002943, 0.549589, 1],[1.356761,-0.012230,0.593896, 129.983047, 198.800766, 1.356761, 1.378454, 0.601491, 0.196810, 0],[1.674299,-0.493095,0.708233, 192.791901, 195.560608, 1.674299, 1.636199, -0.593215, 0.693129, 1],[1.530113,-0.117964,0.574260, 144.960861, 185.067169, 1.530113, 1.526577, -0.220966, 0.566551, 1],[1.542953,-0.144195,0.586389, 148.559830, 186.109573, 1.542953, 1.412656, -0.132459, 0.539291, 0],[1.716656,-0.454253,0.674272, 186.215302, 188.912125, 1.716656, 1.688096, -0.544980, 0.674084, 1],[1.714348,-0.536365,0.671770, 196.831024, 188.707306, 1.714348, 1.480343, 0.007016, 0.580018, 0],[1.677289,-0.381020,0.573245, 177.976151, 177.689072, 1.677289, 1.661456, -0.484436, 0.576379, 1],[1.466350,0.050785,0.563515, 120.380653, 187.045425, 1.466350, 1.266457, -0.199989, 0.537806, 0],[1.459470,0.015009,0.547022, 125.737518, 184.957901, 1.459470, 1.460936, -0.094198, 0.545451, 1],[1.450008,-0.061398,0.552292, 137.315460, 186.295609, 1.450008, 1.258293, 0.500164, 0.224163, 0],[1.621743,-0.274258,0.703413, 165.204926, 197.922546, 1.621743, 1.609692, -0.382419, 0.699255, 1],[1.401586,-0.057354,0.575485, 137.002640, 192.831055, 1.401586, 1.395758, -0.186750, 0.564401, 0],[1.281444,-0.069339,0.549824, 139.904236, 196.894501, 1.281444, 1.524347, 0.036319, 0.543880, 0],[1.689464,-0.369814,0.637302, 176.156784, 185.488663, 1.689464, 1.525658, -0.014759, 0.565350, 0],[1.654714,-0.421554,0.726171, 184.047012, 199.046967, 1.654714, 1.647353, -0.538028, 0.723028, 1],[1.693856,-0.467828,0.733928, 188.762009, 197.823456, 1.693856, 1.519857, -0.259196, 0.589758, 0],[1.449911,0.379905,0.546969, 70.355759, 185.493408, 1.449911, 1.335448, -0.283302, 0.539063, 0],[1.440835,0.393912,0.601902, 67.853935, 194.403931, 1.440835, 1.260732, 0.628060, 0.326933, 0],])
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