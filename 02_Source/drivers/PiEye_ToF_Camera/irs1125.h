/* SPDX-License-Identifier: GPL-2.0 */
/*
 * A V4L2 driver for Infineon IRS1125 TOF cameras.
 * Copyright (C) 2018, pieye GmbH
 *
 * Based on V4L2 OmniVision OV5647 Image Sensor driver
 * Copyright (C) 2016 Ramiro Oliveira <roliveir@synopsys.com>
 *
 * DT / fwnode changes, and GPIO control taken from ov5640.c
 * Copyright (C) 2011-2013 Freescale Semiconductor, Inc. All Rights Reserved.
 * Copyright (C) 2014-2017 Mentor Graphics Inc.
 *
 */

#ifndef IRS1125_H
#define IRS1125_H

#include <linux/v4l2-controls.h>
#include <linux/types.h>

#define IRS1125_NUM_SEQ_ENTRIES 20
#define IRS1125_NUM_MOD_PLLS 4

#define IRS1125_CID_CUSTOM_BASE			(V4L2_CID_USER_BASE | 0xf000)
#define IRS1125_CID_CONTINUOUS_TRIG		(IRS1125_CID_CUSTOM_BASE + 1)
#define IRS1125_CID_TRIGGER			(IRS1125_CID_CUSTOM_BASE + 2)
#define IRS1125_CID_RECONFIG			(IRS1125_CID_CUSTOM_BASE + 3)
#define IRS1125_CID_ILLU_ON			(IRS1125_CID_CUSTOM_BASE + 4)
#define IRS1125_CID_NUM_SEQS			(IRS1125_CID_CUSTOM_BASE + 5)
#define IRS1125_CID_MOD_PLL			(IRS1125_CID_CUSTOM_BASE + 6)
#define IRS1125_CID_SEQ_CONFIG			(IRS1125_CID_CUSTOM_BASE + 7)
#define IRS1125_CID_IDENT0			(IRS1125_CID_CUSTOM_BASE + 8)
#define IRS1125_CID_IDENT1			(IRS1125_CID_CUSTOM_BASE + 9)
#define IRS1125_CID_IDENT2			(IRS1125_CID_CUSTOM_BASE + 10)
#define IRS1125_CID_SAFE_RECONFIG_S0_EXPO	(IRS1125_CID_CUSTOM_BASE + 11)
#define IRS1125_CID_SAFE_RECONFIG_S0_FRAME	(IRS1125_CID_CUSTOM_BASE + 12)
#define IRS1125_CID_SAFE_RECONFIG_S1_EXPO	(IRS1125_CID_CUSTOM_BASE + 13)
#define IRS1125_CID_SAFE_RECONFIG_S1_FRAME	(IRS1125_CID_CUSTOM_BASE + 14)
#define IRS1125_CID_SAFE_RECONFIG_S2_EXPO	(IRS1125_CID_CUSTOM_BASE + 15)
#define IRS1125_CID_SAFE_RECONFIG_S2_FRAME	(IRS1125_CID_CUSTOM_BASE + 16)
#define IRS1125_CID_SAFE_RECONFIG_S3_EXPO	(IRS1125_CID_CUSTOM_BASE + 17)
#define IRS1125_CID_SAFE_RECONFIG_S3_FRAME	(IRS1125_CID_CUSTOM_BASE + 18)
#define IRS1125_CID_SAFE_RECONFIG_S4_EXPO	(IRS1125_CID_CUSTOM_BASE + 19)
#define IRS1125_CID_SAFE_RECONFIG_S4_FRAME	(IRS1125_CID_CUSTOM_BASE + 20)
#define IRS1125_CID_SAFE_RECONFIG_S5_EXPO	(IRS1125_CID_CUSTOM_BASE + 21)
#define IRS1125_CID_SAFE_RECONFIG_S5_FRAME	(IRS1125_CID_CUSTOM_BASE + 22)
#define IRS1125_CID_SAFE_RECONFIG_S6_EXPO	(IRS1125_CID_CUSTOM_BASE + 23)
#define IRS1125_CID_SAFE_RECONFIG_S6_FRAME	(IRS1125_CID_CUSTOM_BASE + 24)
#define IRS1125_CID_SAFE_RECONFIG_S7_EXPO	(IRS1125_CID_CUSTOM_BASE + 25)
#define IRS1125_CID_SAFE_RECONFIG_S7_FRAME	(IRS1125_CID_CUSTOM_BASE + 26)
#define IRS1125_CID_SAFE_RECONFIG_S8_EXPO	(IRS1125_CID_CUSTOM_BASE + 27)
#define IRS1125_CID_SAFE_RECONFIG_S8_FRAME	(IRS1125_CID_CUSTOM_BASE + 28)
#define IRS1125_CID_SAFE_RECONFIG_S9_EXPO	(IRS1125_CID_CUSTOM_BASE + 29)
#define IRS1125_CID_SAFE_RECONFIG_S9_FRAME	(IRS1125_CID_CUSTOM_BASE + 30)
#define IRS1125_CID_SAFE_RECONFIG_S10_EXPO	(IRS1125_CID_CUSTOM_BASE + 31)
#define IRS1125_CID_SAFE_RECONFIG_S10_FRAME	(IRS1125_CID_CUSTOM_BASE + 32)
#define IRS1125_CID_SAFE_RECONFIG_S11_EXPO	(IRS1125_CID_CUSTOM_BASE + 33)
#define IRS1125_CID_SAFE_RECONFIG_S11_FRAME	(IRS1125_CID_CUSTOM_BASE + 34)
#define IRS1125_CID_SAFE_RECONFIG_S12_EXPO	(IRS1125_CID_CUSTOM_BASE + 35)
#define IRS1125_CID_SAFE_RECONFIG_S12_FRAME	(IRS1125_CID_CUSTOM_BASE + 36)
#define IRS1125_CID_SAFE_RECONFIG_S13_EXPO	(IRS1125_CID_CUSTOM_BASE + 37)
#define IRS1125_CID_SAFE_RECONFIG_S13_FRAME	(IRS1125_CID_CUSTOM_BASE + 38)
#define IRS1125_CID_SAFE_RECONFIG_S14_EXPO	(IRS1125_CID_CUSTOM_BASE + 39)
#define IRS1125_CID_SAFE_RECONFIG_S14_FRAME	(IRS1125_CID_CUSTOM_BASE + 40)
#define IRS1125_CID_SAFE_RECONFIG_S15_EXPO	(IRS1125_CID_CUSTOM_BASE + 41)
#define IRS1125_CID_SAFE_RECONFIG_S15_FRAME	(IRS1125_CID_CUSTOM_BASE + 42)
#define IRS1125_CID_SAFE_RECONFIG_S16_EXPO	(IRS1125_CID_CUSTOM_BASE + 43)
#define IRS1125_CID_SAFE_RECONFIG_S16_FRAME	(IRS1125_CID_CUSTOM_BASE + 44)
#define IRS1125_CID_SAFE_RECONFIG_S17_EXPO	(IRS1125_CID_CUSTOM_BASE + 45)
#define IRS1125_CID_SAFE_RECONFIG_S17_FRAME	(IRS1125_CID_CUSTOM_BASE + 46)
#define IRS1125_CID_SAFE_RECONFIG_S18_EXPO	(IRS1125_CID_CUSTOM_BASE + 47)
#define IRS1125_CID_SAFE_RECONFIG_S18_FRAME	(IRS1125_CID_CUSTOM_BASE + 48)
#define IRS1125_CID_SAFE_RECONFIG_S19_EXPO	(IRS1125_CID_CUSTOM_BASE + 49)
#define IRS1125_CID_SAFE_RECONFIG_S19_FRAME	(IRS1125_CID_CUSTOM_BASE + 50)

struct irs1125_seq_cfg {
	__u16 exposure;
	__u16 framerate;
	__u16 ps;
	__u16 pll;
};

struct irs1125_mod_pll {
	__u16 pllcfg1;
	__u16 pllcfg2;
	__u16 pllcfg3;
	__u16 pllcfg4;
	__u16 pllcfg5;
	__u16 pllcfg6;
	__u16 pllcfg7;
	__u16 pllcfg8;
};

#endif /* IRS1125 */
