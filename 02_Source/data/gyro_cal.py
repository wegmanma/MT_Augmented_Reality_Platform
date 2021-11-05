import numpy as np
from matplotlib import pyplot as plt

data = [[0.005479691550136, -0.001842672820203, 0.008161276578903, 1.179478764533997, 4.828158855438232, -8.389670372009277, 1.000000000000000],
        [0.005543119274080, -0.001923113595694, 0.008363617584109,
            8.874924659729004, -1.729482054710388, -2.256148099899292, 1.000000000000000],
        [0.005729134660214, -0.002089996356517, 0.008252636529505, -
            2.502127170562744, -6.552940845489502, 7.127273082733154, 1.000000000000000],
        [0.005447315517813, -0.002219443675131, 0.008332706056535,
        0.419010132551193, -9.559782028198242, 1.917805910110474, 1.000000000000000],
        [0.005426042713225, -0.001859738375060, 0.008291694335639, -
        10.045273780822754, -1.910700321197510, 0.118852324783802, 1.000000000000000],
        [0.005313768517226, -0.001922909752466, 0.008280266076326, -
        5.489676475524902, -8.097599029541016, -1.905570030212402, 1.000000000000000],
        [0.005602704826742, -0.001612102962099, 0.008281549438834, -
        1.447167277336121, 9.230139732360840, -3.641879320144653, 1.000000000000000],
        [0.005513479467481, -0.002229150384665, 0.008174591697752, 9.304054260253906,
        1.015241146087646, -0.422843307256699, 1.000000000000000],
        [0.005434146616608, -0.002531717065722, 0.007107177749276, -
        1.719853758811951, 5.830439567565918, -7.842920303344727, 1.000000000000000],
        [0.005672571714967, -0.002067786641419, 0.008248173631728,
        2.781747579574585, -8.109775543212891, -4.279991149902344, 1.000000000000000],
        [0.005559368524700, -0.001868945197202, 0.008170405402780, 0.241472750902176,
        7.887669563293457, -5.915804862976074, 1.000000000000000],
        [0.005967727862298, -0.002359926700592, 0.008208659477532, -
        2.328973054885864, -6.487227916717529, 7.243680953979492, 1.000000000000000],
        [0.005536390002817, -0.002119859913364, 0.008464710786939,
        5.633830547332764, -7.682883739471436, 0.220138236880302, 1.000000000000000],
        [0.005315782502294, -0.002038633916527, 0.008205706253648,
        9.293101310729980, -0.058358937501907, -1.017847299575806, 1.000000000000000],
        [0.005436039995402, -0.002035493962467, 0.008012288250029, -0.120182812213898,
        10.007036209106445, -0.239682182669640, 1.000000000000000],
        [0.005677079316229, -0.002139809075743, 0.008410115726292, -
        0.231262981891632, 1.819175124168396, 9.942807197570801, 1.000000000000000],
        [0.005602861754596, -0.002159906551242, 0.008225991390646, 0.810410320758820,
        2.479431152343750, -9.373741149902344, 1.000000000000000],
        [0.005767282564193, -0.001812735456042, 0.008269589394331, 0.236862659454346,
        9.442605018615723, -3.056059122085571, 1.000000000000000],
        [0.005572332534939, -0.002004242967814, 0.008197709918022, -
        10.169844627380371, -1.117806911468506, 0.554344773292542, 1.000000000000000],
        [0.005874401889741, -0.001695291721262, 0.008249476552010, -
        5.201097488403320, -8.373361587524414, 1.612373352050781, 1.000000000000000],
        [0.005310653708875, -0.002166057471186, 0.008711449801922,
        5.890059947967529, -6.551208019256592, -3.427675247192383, 1.000000000000000],
        [0.005628775805235, -0.001918674679473, 0.008555800653994,
        6.884011268615723, -6.338593006134033, -1.314905047416687, 1.000000000000000],
        [0.005716585554183, -0.002005537040532, 0.008453230373561,
        5.393778800964355, -1.798297882080078, -7.514432907104492, 1.000000000000000],
        [0.005841267295182, -0.002006426220760, 0.008640092797577, -
        8.142441749572754, 1.244027614593506, -5.997312545776367, 1.000000000000000],
        [0.005574247799814, -0.001900274772197, 0.008463869802654, -
        3.301857948303223, -8.467926979064941, -3.834597826004028, 1.000000000000000],
        [0.004720297642052, -0.002072388539091, 0.008078704588115,
        0.056166905909777, -0.434246391057968, -9.720284461975098, 1.000000000000000],
        [0.005526490975171, -0.002049276372418, 0.008288692682981, -
        1.144612669944763, 9.945672035217285, -0.914807260036469, 1.000000000000000],
        [0.005905117839575, -0.002471889834851, 0.008334847167134, -
        2.338173389434814, -0.190004140138626, 9.868028640747070, 1.000000000000000],
        [0.005947022698820, -0.002428507432342, 0.008331128396094, -
        6.400960922241211, 1.176220297813416, 7.899720191955566, 1.000000000000000],
        [0.005634389352053, -0.002057345816866, 0.008275921456516, -
        3.471186637878418, 6.658219337463379, 6.937315940856934, 1.000000000000000],
        [0.005656214430928, -0.002101739170030, 0.008280740119517, -
        6.654316902160645, 0.084796674549580, 7.760989189147949, 1.000000000000000],
        [0.005788173992187, -0.002201487310231, 0.008282624185085, -
        6.710672378540039, -3.089985132217407, 6.996814250946045, 1.000000000000000],
        [0.005979395005852, -0.002079569734633, 0.008255812339485, -
        6.642739772796631, 5.308278083801270, 5.768566131591797, 1.000000000000000],
        [0.005869272630662, -0.002215744694695, 0.008317978121340, 6.541658401489258,
        6.580166339874268, 2.698088884353638, 1.000000000000000],
        [0.005860868375748, -0.002122959122062, 0.008729930035770, 9.225275039672852,
        1.511816859245300, 1.177864432334900, 1.000000000000000],
        [0.005771028343588, -0.001949781668372, 0.008164417929947,
        8.772556304931641, -3.351956129074097, 0.622324764728546, 1.000000000000000],
        [0.005698265507817, -0.002099647885188, 0.008051088079810, 7.887400150299072,
        5.176475524902344, 1.414136171340942, 1.000000000000000],
        [0.005797480698675, -0.002021577674896, 0.008273226208985, -
        7.751152992248535, -2.404521942138672, 6.171576499938965, 1.000000000000000]
        ]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

print((data[3])[3])
for ii in range(len(data)):
    ax.scatter((data[ii])[3], (data[ii])[4], (data[ii])[5], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
