
#include <iostream>
#include <ctime>
#include <cmath>
#include <stdio.h>
#include <stdlib.h> 
#include <algorithm>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <half.h>
#include <float.h>
#include <string>
#include <sstream>
#include <ImfConvert.h>
#include <math.h>
#include <zlib.h>
#include <ImfRgbaFile.h>
#include <ImfFrameBuffer.h>
#include <ImfChannelList.h>
#include <ImfArray.h>
#include <ImfThreading.h>
#include <IlmThread.h>
#include <Iex.h>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <pthread.h>

HALF_EXPORT const half::uif half::_toFloat[1 << 16] =
#include "toFloat.h"
HALF_EXPORT const unsigned short half::_eLut[1 << 9] =
#include "eLut.h"

#define NUM_THREADS     100
using namespace OPENEXR_IMF_NAMESPACE;
using namespace std;
using namespace IMATH_NAMESPACE;


typedef struct RgbaF
{
	float r;
	float g;
	float b;
	float a;
};

struct kernelOptions
{
	string kernelMode;

	//kernel spatial
	int skernel;
	int sradius;
	double sWeight;
	double sColor;
	double sAlbedo;
	double sNormal;
	double sDepth;
	double sAlpha;
	double stColor;
	double stAlbedo;
	double stNormal;
	double stDepth;
	double stAlpha;
	double sFallof;
	double albedoTreshold;
	bool albedoDivide;
	double sSpecularStrength;

	//kernel spatial final touch
	int sfkernel;
	int sfradius;
	double sfWeight;
	double sfColor;
	double sfAlbedo;
	double sfNormal;
	double sfDepth;
	double sfFallof;
	bool sfAlbedoDivide;
	double sfSpecularStrength;

	//kernel temporal
	double temporalWeight;
	int tkernel;
	int tPRadius;
	double tFallof;
	int tBlockSize;
	double tInterpolation;
	double tSigmaColor;
	double tSigmaAlbedo;
	double tMotionTreshold;
	double tColorTreshold;

	//kernel PWtemporal
	double temporalPwWeight;
	int tPwKernel;
	int tPwPRadius;
	double tPwFallof;
	int tPwSearchRadius;
	double tPwSigmaColor;
	double tPwSigmaAlbedo;
	double tPwSigmaDistance;
	double tPwSpaceTreshold;
	int tPwIterations;

	//Firefly Filter
	int ffkernel;
	double ffSigma;
	double ffGain;
	double ffGamma;
	double ffRefractionStrange;
	double ffindirectSpecStrange;
};

struct imageOptions
{
	string runMode;
	string exrlayers;
	int runBlock;
	int startFrame;
	int endFrame;
	int nFrames;
	int nCores;
	char *oPostfix;
	char *mvType;
	double mvScale;
	int nChannels;
	int with;
	int height;
	Array<int> existsFrame;
	char *fnameBeauty;
	char *fnameAlbedo;
	char *fnameNormal;
	char *fnamePosition;
	char *fnameDepth;
	char *fnameDiffuse;
	char *fnameSpecular;
	char *fnameIndirectDiffuse;
	char *fnameIndirectSpecular;
	char *fnameRefraction;
	char *fnameChannels[100];
};

struct imgPixels
{
	Array<Array2D<RgbaF>> pixelsBeauty;
	Array<Array2D<RgbaF>> pixelsAlbedo;
	Array<Array2D<RgbaF>> pixelsNormal;
	Array<Array2D<RgbaF>> pixelsPosition;
	Array<Array2D<RgbaF>> pixelsDepth;
	Array<Array2D<RgbaF>> pixelsDiffuse;
	Array<Array2D<RgbaF>> pixelsIndirectDiffuse;
	Array<Array2D<RgbaF>> pixelsIndirectSpecular;
	Array<Array2D<RgbaF>> pixelsSpecular;
	Array<Array2D<RgbaF>> pixelsRefraction;
};

Array2D<Rgba> tSWeight(0, 0);
Array2D<Rgba> tDWeight(0, 0);
Array2D<Rgba> FireflyWeight(0, 0);
Array2D<Rgba> pix_res(0, 0);
Array2D<Rgba> pix_resT(0, 0);
Array2D<Rgba> pix_resSpecular(0, 0);
Array2D<Rgba> pix_resAlbedo(0, 0);
Array2D<Rgba> pix_resDiffuse(0, 0);
Array2D<Rgba> pix_resIndDiffuse(0, 0);
Array2D<Rgba> pix_resIndSpecular(0, 0);
Array2D<Rgba> pix_resRefraction(0, 0);
Array<Array2D<Rgba>> blockMV(0);
Array<Array2D<Rgba>> blockMVPosition(0);
Array<Array2D<RgbaF>> ResultBlockSmoothPosition(0);

kernelOptions kernelOpt;
imageOptions iOpt;

imgPixels pixels;
imgPixels ResultBlockSmooth;
imgPixels ResultBlockSmoothTemporal;
imgPixels ResultBlockSmoothTMP;
imgPixels ResultBlockSmoothTemporalTMP;

void setKernelPreset(std::string presetName, kernelOptions &kernelOpt)
{
	if (presetName == "default") {
		kernelOpt.sColor = 0.4;
		kernelOpt.sAlbedo = 0.1;
		kernelOpt.sNormal = 0.1;
		kernelOpt.sDepth = 30;
		kernelOpt.sAlpha = -1;
		kernelOpt.stAlpha = -1;
		kernelOpt.tPRadius = 0;
		kernelOpt.tPwPRadius = 0;
		kernelOpt.tColorTreshold = 1;
		kernelOpt.temporalWeight = 1;
		kernelOpt.albedoTreshold = 0.001;
		kernelOpt.albedoDivide = true;
	}
	else
		if (presetName == "staticOneFrame") {
			kernelOpt.sColor = 0.4;
			kernelOpt.sAlbedo = 0.1;
			kernelOpt.sNormal = 0.1;
			kernelOpt.sDepth = 4;
			kernelOpt.tColorTreshold = 1;
			kernelOpt.temporalWeight = 0;
			kernelOpt.albedoTreshold = 0.05;
			kernelOpt.albedoDivide = true;
		}
		else
			if (presetName == "chars") {
				kernelOpt.sColor = 1;
				kernelOpt.sAlbedo = 0.08;
				kernelOpt.sNormal = 0.12;
				kernelOpt.sDepth = 100;
				kernelOpt.tColorTreshold = 1;
				kernelOpt.temporalWeight = 0;
				kernelOpt.albedoTreshold = 0.3;
				kernelOpt.albedoDivide = true;
			}
}

//help message
void helpMessage(std::string name)
{
	std::cerr << "Usage: " << name << " [ -h | -n <numberOfImages> | -b <blur> | -k <opt.kernelWidth> | -c <contribution> | -i <imageSequence> | -o <output> ]" << std::endl
		<< "Options:" << std::endl
		<< "\t-h, --help\t\tShow this help message." << std::endl
		<< "\t-o, --output X\t\tSpecifies the output path." << std::endl
		<< "\t-i, --image X\t\tChange the preset sequence of images to filter." << std::endl
		<< std::endl;
}

bool initializeOptions(int argc, char* argv[], imageOptions &iOpt, kernelOptions &kernelOpt)
{
	//default
	iOpt.mvType = "PRMAN";
	iOpt.mvScale = 1;
	iOpt.startFrame = 1;
	iOpt.endFrame = 1;
	iOpt.nFrames = 1;
	iOpt.nCores = 1;
	iOpt.nChannels = 0;
	iOpt.fnameAlbedo = "";
	iOpt.fnameBeauty = "";
	for (int i = 0; i<100; i++)
		iOpt.fnameChannels[i] = "";
	iOpt.fnameDepth = "";
	iOpt.fnameNormal = "";
	iOpt.fnamePosition = "";
	iOpt.fnameDiffuse = "";
	iOpt.fnameIndirectDiffuse = "";
	iOpt.fnameIndirectSpecular = "";
	iOpt.fnameSpecular = "";
	iOpt.fnameRefraction = "";
	iOpt.with = 0;
	iOpt.height = 0;
	iOpt.existsFrame.resizeErase(iOpt.nFrames);
	setKernelPreset("default", kernelOpt);
	for (int i = 0; i < argc; i++) {
		std::string arg = argv[i];
		if ((arg == "-h") || (arg == "--help"))
		{
			helpMessage(argv[0]);
			return 0;
		}
		else
			if ((arg == "-nf") || (arg == "--numberFrames"))
			{
				if (i + 1 < argc)
				{
					iOpt.nFrames = std::atoi(argv[++i]);
					iOpt.existsFrame.resizeErase(iOpt.nFrames);
					if (iOpt.nFrames < 1) {
						iOpt.nFrames = 1;
						iOpt.existsFrame.resizeErase(iOpt.nFrames);
					}
					if (iOpt.nFrames > 11)
					{
						iOpt.nFrames = 11;
						iOpt.existsFrame.resizeErase(iOpt.nFrames);
						std::cerr << "There are only 11 images available per frame. Be aware that multiple frames will be reused." << std::endl;
					}
				}
				else
				{
					std::cerr << "--numberFrames option requires one argument." << std::endl;
					return 0;
				}
			}
			else
				if ((arg == "-s") || (arg == "--startFrame"))
				{
					if (i + 1 < argc)
					{
						iOpt.startFrame = std::atoi(argv[++i]);
						if (iOpt.startFrame < 0)
							iOpt.startFrame = 0;
					}
					else
					{
						std::cerr << "--startFrame option requires one argument." << std::endl;
						return 0;
					}
				}
				else
					if ((arg == "-e") || (arg == "--endFrame"))
					{
						if (i + 1 < argc)
						{
							iOpt.endFrame = std::atoi(argv[++i]);
							if (iOpt.endFrame < 0)
								iOpt.endFrame = 0;
						}
						else
						{
							std::cerr << "--endFrame option requires one argument." << std::endl;
							return 0;
						}
					}
					else
						if ((arg == "-runBlock") || (arg == "--runBlock"))
						{
							if (i + 1 < argc)
							{
								iOpt.runBlock = std::atoi(argv[++i]);
								if (iOpt.runBlock < 0)
									iOpt.runBlock = 0;
							}
							else
							{
								std::cerr << "--runBlock option requires one argument." << std::endl;
								return 0;
							}
						}
						else
							if ((arg == "-mvt") || (arg == "--mvType"))
							{
								if (i + 1 < argc)
								{
									iOpt.mvType = argv[++i];
								}
								else
								{
									std::cerr << "--mvType option requires one argument." << std::endl;
									return 0;
								}
							}
							else
								if ((arg == "-runMode"))
								{
									if (i + 1 < argc)
									{
										iOpt.runMode = argv[++i];
									}
									else
									{
										std::cerr << "-runMode option requires one argument." << std::endl;
										return 0;
									}
								}
								else
									if ((arg == "-fmode") || (arg == "--filterMode"))
									{
										if (i + 1 < argc)
										{
											kernelOpt.kernelMode = argv[++i];
										}
										else
										{
											std::cerr << "--filterMode option requires one argument." << std::endl;
											return 0;
										}
									}
									else
										if ((arg == "-exrlayers") || (arg == "--exrlayers"))
										{
											if (i + 1 < argc)
											{
												iOpt.exrlayers = argv[++i];
											}
											else
											{
												std::cerr << "--ExrLayers option requires one argument." << std::endl;
												return 0;
											}
										}
										else
											if ((arg == "-ncores") || (arg == "--numberCores"))
											{
												if (i + 1 < argc)
												{
													iOpt.nCores = std::atoi(argv[++i]);
												}
												else
												{
													std::cerr << "--numberCores option requires one argument." << std::endl;
													return 0;
												}
											}
											else
												if ((arg == "-oPostfix") || (arg == "--OutPostfix"))
												{
													if (i + 1 < argc)
													{
														iOpt.oPostfix = argv[++i];
													}
													else
													{
														std::cerr << "--OutPostfix option requires one argument." << std::endl;
														return 0;
													}
												}
												else
													if ((arg == "-mvs") || (arg == "--mvScale"))
													{
														if (i + 1 < argc)
														{
															iOpt.mvScale = std::atof(argv[++i]);
														}
														else
														{
															std::cerr << "--mvScale option requires one argument." << std::endl;
															return 0;
														}
													}
													else
														if ((arg == "-kp") || (arg == "--kernelPreset"))
														{
															if (i + 1 < argc)
															{
																setKernelPreset(argv[++i], kernelOpt);

															}
															else
															{
																std::cerr << "--kernelPreset option requires one argument." << std::endl;
																return 0;
															}
														}
														else
															if ((arg == "-bc") || (arg == "--beautyChannel"))
															{
																if (i + 1 < argc)
																{
																	iOpt.fnameBeauty = argv[++i];
																}
																else
																{
																	std::cerr << "--beautyChannel option requires one argument." << std::endl;
																	return 0;
																}

															}
															else
																if ((arg == "-nc") || (arg == "--normalChannel"))
																{
																	if (i + 1 < argc)
																	{
																		iOpt.fnameNormal = argv[++i];
																	}
																	else
																	{
																		std::cerr << "--normalChannel option requires one argument." << std::endl;
																		return 0;
																	}
																}
																else
																	if ((arg == "-ac") || (arg == "--albedoChannel"))
																	{
																		if (i + 1 < argc)
																		{
																			iOpt.fnameAlbedo = argv[++i];
																		}
																		else
																		{
																			std::cerr << "--albedoChannel option requires one argument." << std::endl;
																			return 0;
																		}
																	}
																	else
																		if ((arg == "-pc") || (arg == "--positionChannel"))
																		{
																			if (i + 1 < argc)
																			{
																				iOpt.fnamePosition = argv[++i];
																			}
																			else
																			{
																				std::cerr << "--positionChannel option requires one argument." << std::endl;
																				return 0;
																			}
																		}
																		else
																			if ((arg == "-dc") || (arg == "--DepthChannel"))
																			{
																				if (i + 1 < argc)
																				{
																					iOpt.fnameDepth = argv[++i];
																				}
																				else
																				{
																					std::cerr << "--DepthChannel option requires one argument." << std::endl;
																					return 0;
																				}
																			}
																			else
																				if ((arg == "-ch") || (arg == "--Channels"))
																				{
																					if (i + 1 < argc)
																					{
																						for (int nch = i + 1; nch < argc; nch++) {
																							iOpt.fnameChannels[nch - i - 1] = argv[nch];
																						}
																						iOpt.nChannels = argc - i - 1;
																					}
																					else
																					{
																						std::cerr << "--Channels option requires one argument." << std::endl;
																						return 0;
																					}
																				}
																				else
																					if ((arg == "-sc") || (arg == "--SpecularChannel"))
																					{
																						if (i + 1 < argc)
																						{
																							iOpt.fnameSpecular = argv[++i];
																						}
																						else
																						{
																							std::cerr << "--SpecularChannel option requires one argument." << std::endl;
																							return 0;
																						}
																					}
																					else
																						if ((arg == "-isc") || (arg == "--IndirectSpecularChannel"))
																						{
																							if (i + 1 < argc)
																							{
																								iOpt.fnameIndirectSpecular = argv[++i];
																							}
																							else
																							{
																								std::cerr << "--IndirectSpecularChannel option requires one argument." << std::endl;
																								return 0;
																							}
																						}
																						else
																							if ((arg == "-dic") || (arg == "--DiffuseChannel"))
																							{
																								if (i + 1 < argc)
																								{
																									iOpt.fnameDiffuse = argv[++i];
																								}
																								else
																								{
																									std::cerr << "--DiffuseChannel option requires one argument." << std::endl;
																									return 0;
																								}
																							}
																							else
																								if ((arg == "-idic") || (arg == "--IndirectDiffuseChannel"))
																								{
																									if (i + 1 < argc)
																									{
																										iOpt.fnameIndirectDiffuse = argv[++i];
																									}
																									else
																									{
																										std::cerr << "--IndirectDiffuseChannel option requires one argument." << std::endl;
																										return 0;
																									}
																								}
																								else
																									if ((arg == "-rc") || (arg == "--RefractionChannel"))
																									{
																										if (i + 1 < argc)
																										{
																											iOpt.fnameRefraction = argv[++i];
																										}
																										else
																										{
																											std::cerr << "--RefractionDiffuseChannel option requires one argument." << std::endl;
																											return 0;
																										}
																									}
																									else
																										if ((arg == "-ffkernel") || (arg == "--fireflyKernel"))
																										{
																											if (i + 1 < argc)
																											{
																												kernelOpt.ffkernel = std::atoi(argv[++i]);

																											}
																											else
																											{
																												std::cerr << "--fireflyKernel option requires one argument." << std::endl;
																												return 0;
																											}
																										}
																										else
																											if ((arg == "-ffgain") || (arg == "--fireflyGain"))
																											{
																												if (i + 1 < argc)
																												{
																													kernelOpt.ffGain = std::atof(argv[++i]);

																												}
																												else
																												{
																													std::cerr << "--fireflyGain option requires one argument." << std::endl;
																													return 0;
																												}
																											}
																											else
																												if ((arg == "-ffsigma") || (arg == "--fireflySigma"))
																												{
																													if (i + 1 < argc)
																													{
																														kernelOpt.ffSigma = std::atof(argv[++i]);

																													}
																													else
																													{
																														std::cerr << "--fireflySigma option requires one argument." << std::endl;
																														return 0;
																													}
																												}
																												else
																													if ((arg == "-ffgamma") || (arg == "--fireflyGamma"))
																													{
																														if (i + 1 < argc)
																														{
																															kernelOpt.ffGamma = std::atof(argv[++i]);
																														}
																														else
																														{
																															std::cerr << "--fireflyGamma option requires one argument." << std::endl;
																															return 0;
																														}
																													}
																													else
																														if ((arg == "-ffRefractionStrange") || (arg == "--ffRefractionStrange"))
																														{
																															if (i + 1 < argc)
																															{
																																kernelOpt.ffRefractionStrange = std::atof(argv[++i]);
																															}
																															else
																															{
																																std::cerr << "--ffRefractionStrange option requires one argument." << std::endl;
																																return 0;
																															}
																														}
																														else
																															if ((arg == "-ffIndirectSpecularStrange") || (arg == "--ffIndirectSpecularStrange"))
																															{
																																if (i + 1 < argc)
																																{
																																	kernelOpt.ffindirectSpecStrange = std::atof(argv[++i]);
																																}
																																else
																																{
																																	std::cerr << "--ffIndirectSpecularStrange option requires one argument." << std::endl;
																																	return 0;
																																}
																															}
																															else
																																if ((arg == "-fsKernel") || (arg == "--filterSKernel"))
																																{
																																	if (i + 1 < argc)
																																	{
																																		kernelOpt.skernel = std::atoi(argv[++i]);
																																	}
																																	else
																																	{
																																		std::cerr << "--filterSKernel option requires one argument." << std::endl;
																																		return 0;
																																	}
																																}
																																else
																																	if ((arg == "-fsRadius") || (arg == "--filterSRadius"))
																																	{
																																		if (i + 1 < argc)
																																		{
																																			kernelOpt.sradius = std::atoi(argv[++i]);
																																		}
																																		else
																																		{
																																			std::cerr << "--filterSRadius option requires one argument." << std::endl;
																																			return 0;
																																		}
																																	}
																																	else
																																		if ((arg == "-fsSigmaColor") || (arg == "--filterSSigmaColor"))
																																		{
																																			if (i + 1 < argc)
																																			{
																																				kernelOpt.sColor = std::atof(argv[++i]);
																																			}
																																			else
																																			{
																																				std::cerr << "--filterSSigmaColor option requires one argument." << std::endl;
																																				return 0;
																																			}
																																		}
																																		else
																																			if ((arg == "-fsSigmaAlbedo") || (arg == "--filterSSigmaAlbedo"))
																																			{
																																				if (i + 1 < argc)
																																				{
																																					kernelOpt.sAlbedo = std::atof(argv[++i]);
																																				}
																																				else
																																				{
																																					std::cerr << "--filterSSigmaAlbedo option requires one argument." << std::endl;
																																					return 0;
																																				}
																																			}
																																			else
																																				if ((arg == "-fsSigmaNormal") || (arg == "--filterSSigmaNormal"))
																																				{
																																					if (i + 1 < argc)
																																					{
																																						kernelOpt.sNormal = std::atof(argv[++i]);
																																					}
																																					else
																																					{
																																						std::cerr << "--filterSSigmaNormal option requires one argument." << std::endl;
																																						return 0;
																																					}
																																				}
																																				else
																																					if ((arg == "-fsSigmaDepth") || (arg == "--filterSSigmaDepth"))
																																					{
																																						if (i + 1 < argc)
																																						{
																																							kernelOpt.sDepth = std::atof(argv[++i]);
																																						}
																																						else
																																						{
																																							std::cerr << "--filterSSigmaDepth option requires one argument." << std::endl;
																																							return 0;
																																						}
																																					}
																																					else
																																						if ((arg == "-fsSigmaAlpha") || (arg == "--filterSSigmaAlpha"))
																																						{
																																							if (i + 1 < argc)
																																							{
																																								kernelOpt.sAlpha = std::atof(argv[++i]);
																																							}
																																							else
																																							{
																																								std::cerr << "--filterSSigmaAlpha option requires one argument." << std::endl;
																																								return 0;
																																							}
																																						}
																																						else
																																							if ((arg == "-fstSigmaColor") || (arg == "--filterSTSigmaColor"))
																																							{
																																								if (i + 1 < argc)
																																								{
																																									kernelOpt.stColor = std::atof(argv[++i]);
																																								}
																																								else
																																								{
																																									std::cerr << "--filterSTSigmaColor option requires one argument." << std::endl;
																																									return 0;
																																								}
																																							}
																																							else
																																								if ((arg == "-fstSigmaAlbedo") || (arg == "--filterSTSigmaAlbedo"))
																																								{
																																									if (i + 1 < argc)
																																									{
																																										kernelOpt.stAlbedo = std::atof(argv[++i]);
																																									}
																																									else
																																									{
																																										std::cerr << "--filterSTSigmaAlbedo option requires one argument." << std::endl;
																																										return 0;
																																									}
																																								}
																																								else
																																									if ((arg == "-fstSigmaNormal") || (arg == "--filterSTSigmaNormal"))
																																									{
																																										if (i + 1 < argc)
																																										{
																																											kernelOpt.stNormal = std::atof(argv[++i]);
																																										}
																																										else
																																										{
																																											std::cerr << "--filterSTSigmaNormal option requires one argument." << std::endl;
																																											return 0;
																																										}
																																									}
																																									else
																																										if ((arg == "-fstSigmaDepth") || (arg == "--filterSTSigmaDepth"))
																																										{
																																											if (i + 1 < argc)
																																											{
																																												kernelOpt.stDepth = std::atof(argv[++i]);
																																											}
																																											else
																																											{
																																												std::cerr << "--filterSTSigmaDepth option requires one argument." << std::endl;
																																												return 0;
																																											}
																																										}
																																										else
																																											if ((arg == "-fstSigmaAlpha") || (arg == "--filterSTSigmaAlpha"))
																																											{
																																												if (i + 1 < argc)
																																												{
																																													kernelOpt.stAlpha = std::atof(argv[++i]);
																																												}
																																												else
																																												{
																																													std::cerr << "--filterSTSigmaAlpha option requires one argument." << std::endl;
																																													return 0;
																																												}
																																											}
																																											else
																																												if ((arg == "-fsFallof") || (arg == "--filterSFallof"))
																																												{
																																													if (i + 1 < argc)
																																													{
																																														kernelOpt.sFallof = std::atof(argv[++i]);
																																													}
																																													else
																																													{
																																														std::cerr << "--filterSFallof option requires one argument." << std::endl;
																																														return 0;
																																													}
																																												}
																																												else
																																													if ((arg == "-fsw") || (arg == "--filterSWeight"))
																																													{
																																														if (i + 1 < argc)
																																														{
																																															kernelOpt.sWeight = std::atof(argv[++i]);
																																															if (kernelOpt.sWeight > 1) {
																																																kernelOpt.sWeight = 1;
																																															}
																																															if (kernelOpt.sWeight < 0) {
																																																kernelOpt.sWeight = 0;
																																															}
																																														}
																																														else
																																														{
																																															std::cerr << "--filterSWeight option requires one argument." << std::endl;
																																															return 0;
																																														}
																																													}
																																													else
																																														if ((arg == "-fse") || (arg == "--filterSEpsilon"))
																																														{
																																															if (i + 1 < argc)
																																															{
																																																kernelOpt.albedoTreshold = std::atof(argv[++i]);
																																															}
																																															else
																																															{
																																																std::cerr << "--filterSEpsilon option requires one argument." << std::endl;
																																																return 0;
																																															}
																																														}
																																														else
																																															if ((arg == "-fsad") || (arg == "--filterSAlbedoDivide"))
																																															{
																																																if (i + 1 < argc)
																																																{
																																																	kernelOpt.albedoDivide = std::atoi(argv[++i]);
																																																}
																																																else
																																																{
																																																	std::cerr << "--filterSAlbedoDivide option requires one argument." << std::endl;
																																																	return 0;
																																																}
																																															}
																																															else
																																																if ((arg == "-fsSpecularStrength") || (arg == "--fsSpecularStrength"))
																																																{
																																																	if (i + 1 < argc)
																																																	{
																																																		kernelOpt.sSpecularStrength = std::atof(argv[++i]);
																																																	}
																																																	else
																																																	{
																																																		std::cerr << "--fsSpecularStrength option requires one argument." << std::endl;
																																																		return 0;
																																																	}
																																																}
																																																else
																																																	if ((arg == "-fsfKernel") || (arg == "--filterSFKernel"))
																																																	{
																																																		if (i + 1 < argc)
																																																		{
																																																			kernelOpt.sfkernel = std::atoi(argv[++i]);
																																																		}
																																																		else
																																																		{
																																																			std::cerr << "--filterSFKernel option requires one argument." << std::endl;
																																																			return 0;
																																																		}
																																																	}
																																																	else
																																																		if ((arg == "-fsfRadius") || (arg == "--filterSFRadius"))
																																																		{
																																																			if (i + 1 < argc)
																																																			{
																																																				kernelOpt.sfradius = std::atoi(argv[++i]);
																																																			}
																																																			else
																																																			{
																																																				std::cerr << "--filterSFRadius option requires one argument." << std::endl;
																																																				return 0;
																																																			}
																																																		}
																																																		else
																																																			if ((arg == "-fsfSigmaColor") || (arg == "--filterSFSigmaColor"))
																																																			{
																																																				if (i + 1 < argc)
																																																				{
																																																					kernelOpt.sfColor = std::atof(argv[++i]);
																																																				}
																																																				else
																																																				{
																																																					std::cerr << "--filterSFSigmaColor option requires one argument." << std::endl;
																																																					return 0;
																																																				}
																																																			}
																																																			else
																																																				if ((arg == "-fsfSigmaAlbedo") || (arg == "--filterSFSigmaAlbedo"))
																																																				{
																																																					if (i + 1 < argc)
																																																					{
																																																						kernelOpt.sfAlbedo = std::atof(argv[++i]);
																																																					}
																																																					else
																																																					{
																																																						std::cerr << "--filterSFSigmaAlbedo option requires one argument." << std::endl;
																																																						return 0;
																																																					}
																																																				}
																																																				else
																																																					if ((arg == "-fsfSigmaNormal") || (arg == "--filterSFSigmaNormal"))
																																																					{
																																																						if (i + 1 < argc)
																																																						{
																																																							kernelOpt.sfNormal = std::atof(argv[++i]);
																																																						}
																																																						else
																																																						{
																																																							std::cerr << "--filterSFSigmaNormal option requires one argument." << std::endl;
																																																							return 0;
																																																						}
																																																					}
																																																					else
																																																						if ((arg == "-fsfSigmaDepth") || (arg == "--filterSFSigmaDepth"))
																																																						{
																																																							if (i + 1 < argc)
																																																							{
																																																								kernelOpt.sfDepth = std::atof(argv[++i]);
																																																							}
																																																							else
																																																							{
																																																								std::cerr << "--filterSFSigmaDepth option requires one argument." << std::endl;
																																																								return 0;
																																																							}
																																																						}
																																																						else
																																																							if ((arg == "-fsfFallof") || (arg == "--filterSFFallof"))
																																																							{
																																																								if (i + 1 < argc)
																																																								{
																																																									kernelOpt.sfFallof = std::atof(argv[++i]);
																																																								}
																																																								else
																																																								{
																																																									std::cerr << "--filterSFFallof option requires one argument." << std::endl;
																																																									return 0;
																																																								}
																																																							}
																																																							else
																																																								if ((arg == "-fsfw") || (arg == "--filterSFWeight"))
																																																								{
																																																									if (i + 1 < argc)
																																																									{
																																																										kernelOpt.sfWeight = std::atof(argv[++i]);
																																																										if (kernelOpt.sfWeight > 1) {
																																																											kernelOpt.sfWeight = 1;
																																																										}
																																																										if (kernelOpt.sfWeight < 0) {
																																																											kernelOpt.sfWeight = 0;
																																																										}
																																																									}
																																																									else
																																																									{
																																																										std::cerr << "--filterSFWeight option requires one argument." << std::endl;
																																																										return 0;
																																																									}
																																																								}
																																																								else
																																																									if ((arg == "-fsfad") || (arg == "--filterSFAlbedoDivide"))
																																																									{
																																																										if (i + 1 < argc)
																																																										{
																																																											kernelOpt.sfAlbedoDivide = std::atoi(argv[++i]);
																																																										}
																																																										else
																																																										{
																																																											std::cerr << "--filterSAlbedoDivide option requires one argument." << std::endl;
																																																											return 0;
																																																										}
																																																									}
																																																									else
																																																										if ((arg == "-fsfSpecularStrength") || (arg == "--fsfSpecularStrength"))
																																																										{
																																																											if (i + 1 < argc)
																																																											{
																																																												kernelOpt.sfSpecularStrength = std::atof(argv[++i]);
																																																											}
																																																											else
																																																											{
																																																												std::cerr << "--fsfSpecularStrength option requires one argument." << std::endl;
																																																												return 0;
																																																											}
																																																										}
																																																										else
																																																											if ((arg == "-ftw") || (arg == "--filterTWeight"))
																																																											{
																																																												if (i + 1 < argc)
																																																												{
																																																													kernelOpt.temporalWeight = std::atof(argv[++i]);
																																																													if (kernelOpt.temporalWeight > 1) {
																																																														kernelOpt.temporalWeight = 1;
																																																													}
																																																													if (kernelOpt.temporalWeight < 0) {
																																																														kernelOpt.temporalWeight = 0;
																																																													}
																																																												}
																																																												else
																																																												{
																																																													std::cerr << "--filterTWeight option requires one argument." << std::endl;
																																																													return 0;
																																																												}
																																																											}
																																																											else
																																																												if ((arg == "-ftFallof") || (arg == "--filterTFallof"))
																																																												{
																																																													if (i + 1 < argc)
																																																													{
																																																														kernelOpt.tFallof = std::atof(argv[++i]);
																																																													}
																																																													else
																																																													{
																																																														std::cerr << "--filterTFallof option requires one argument." << std::endl;
																																																														return 0;
																																																													}
																																																												}
																																																												else
																																																													if ((arg == "-ftKernel") || (arg == "--filterTKernel"))
																																																													{
																																																														if (i + 1 < argc)
																																																														{
																																																															kernelOpt.tkernel = std::atoi(argv[++i]);
																																																														}
																																																														else
																																																														{
																																																															std::cerr << "--filterTKernel option requires one argument." << std::endl;
																																																															return 0;
																																																														}
																																																													}
																																																													else
																																																														if ((arg == "-ftPRadius") || (arg == "--filterTPRadius"))
																																																														{
																																																															if (i + 1 < argc)
																																																															{
																																																																kernelOpt.tPRadius = std::atoi(argv[++i]);
																																																															}
																																																															else
																																																															{
																																																																std::cerr << "--filterTPRadius option requires one argument." << std::endl;
																																																																return 0;
																																																															}
																																																														}
																																																														else
																																																															if ((arg == "-ftbs") || (arg == "--filterTBlockSize"))
																																																															{
																																																																if (i + 1 < argc)
																																																																{
																																																																	kernelOpt.tBlockSize = std::atoi(argv[++i]);
																																																																}
																																																																else
																																																																{
																																																																	std::cerr << "--filterTBlockSize option requires one argument." << std::endl;
																																																																	return 0;
																																																																}
																																																															}
																																																															else
																																																																if ((arg == "-ftISize") || (arg == "--filterTInterpolation"))
																																																																{
																																																																	if (i + 1 < argc)
																																																																	{
																																																																		kernelOpt.tInterpolation = std::atof(argv[++i]);
																																																																	}
																																																																	else
																																																																	{
																																																																		std::cerr << "--filterTInterpolation option requires one argument." << std::endl;
																																																																		return 0;
																																																																	}
																																																																}
																																																																else
																																																																	if ((arg == "-ftSigmaColor") || (arg == "--filterTSigmaColor"))
																																																																	{
																																																																		if (i + 1 < argc)
																																																																		{
																																																																			kernelOpt.tSigmaColor = std::atof(argv[++i]);
																																																																		}
																																																																		else
																																																																		{
																																																																			std::cerr << "--filterTSigmaColor option requires one argument." << std::endl;
																																																																			return 0;
																																																																		}
																																																																	}
																																																																	else
																																																																		if ((arg == "-ftSigmaAlbedo") || (arg == "--filterTSigmaAlbedo"))
																																																																		{
																																																																			if (i + 1 < argc)
																																																																			{
																																																																				kernelOpt.tSigmaAlbedo = std::atof(argv[++i]);
																																																																			}
																																																																			else
																																																																			{
																																																																				std::cerr << "--filterTSigmaColor option requires one argument." << std::endl;
																																																																				return 0;
																																																																			}
																																																																		}
																																																																		else
																																																																			if ((arg == "-ftmt") || (arg == "--filterTMotionTreshold"))
																																																																			{
																																																																				if (i + 1 < argc)
																																																																				{
																																																																					kernelOpt.tMotionTreshold = std::atof(argv[++i]);
																																																																				}
																																																																				else
																																																																				{
																																																																					std::cerr << "--filterTMotionTreshold option requires one argument." << std::endl;
																																																																					return 0;
																																																																				}
																																																																			}
																																																																			else
																																																																				if ((arg == "-ftct") || (arg == "--filterTColorTreshold"))
																																																																				{
																																																																					if (i + 1 < argc)
																																																																					{
																																																																						kernelOpt.tColorTreshold = std::atof(argv[++i]);
																																																																					}
																																																																					else
																																																																					{
																																																																						std::cerr << "--filterTColorTreshold option requires one argument." << std::endl;
																																																																						return 0;
																																																																					}
																																																																				}
																																																																				else
																																																																					if ((arg == "-ftpww") || (arg == "--filterTPwWeight"))
																																																																					{
																																																																						if (i + 1 < argc)
																																																																						{
																																																																							kernelOpt.temporalPwWeight = std::atof(argv[++i]);
																																																																							if (kernelOpt.temporalPwWeight > 1) {
																																																																								kernelOpt.temporalPwWeight = 1;
																																																																							}
																																																																							if (kernelOpt.temporalPwWeight < 0) {
																																																																								kernelOpt.temporalPwWeight = 0;
																																																																							}
																																																																						}
																																																																						else
																																																																						{
																																																																							std::cerr << "--filterTPwWeight option requires one argument." << std::endl;
																																																																							return 0;
																																																																						}
																																																																					}
																																																																					else
																																																																						if ((arg == "-ftpwFallof") || (arg == "--filterTPwFallof"))
																																																																						{
																																																																							if (i + 1 < argc)
																																																																							{
																																																																								kernelOpt.tPwFallof = std::atof(argv[++i]);
																																																																							}
																																																																							else
																																																																							{
																																																																								std::cerr << "--filterTPwFallof option requires one argument." << std::endl;
																																																																								return 0;
																																																																							}
																																																																						}
																																																																						else
																																																																							if ((arg == "-ftpwKernel") || (arg == "--filterTPwKernel"))
																																																																							{
																																																																								if (i + 1 < argc)
																																																																								{
																																																																									kernelOpt.tPwKernel = std::atoi(argv[++i]);
																																																																								}
																																																																								else
																																																																								{
																																																																									std::cerr << "--filterTPwKernel option requires one argument." << std::endl;
																																																																									return 0;
																																																																								}
																																																																							}
																																																																							else
																																																																								if ((arg == "-ftpwPRadius") || (arg == "--filterTPwPRadius"))
																																																																								{
																																																																									if (i + 1 < argc)
																																																																									{
																																																																										kernelOpt.tPwPRadius = std::atoi(argv[++i]);
																																																																									}
																																																																									else
																																																																									{
																																																																										std::cerr << "--filterTPwPRadius option requires one argument." << std::endl;
																																																																										return 0;
																																																																									}
																																																																								}
																																																																								else
																																																																									if ((arg == "-ftpwSRadius") || (arg == "--filterPwSearchRadius"))
																																																																									{
																																																																										if (i + 1 < argc)
																																																																										{
																																																																											kernelOpt.tPwSearchRadius = std::atoi(argv[++i]);
																																																																										}
																																																																										else
																																																																										{
																																																																											std::cerr << "--filterPwSearchRadius option requires one argument." << std::endl;
																																																																											return 0;
																																																																										}
																																																																									}
																																																																									else
																																																																										if ((arg == "-ftpwSigmaColor") || (arg == "--filterTPwSigmaColor"))
																																																																										{
																																																																											if (i + 1 < argc)
																																																																											{
																																																																												kernelOpt.tPwSigmaColor = std::atof(argv[++i]);
																																																																											}
																																																																											else
																																																																											{
																																																																												std::cerr << "--filterTPwSigmaColor option requires one argument." << std::endl;
																																																																												return 0;
																																																																											}
																																																																										}
																																																																										else
																																																																											if ((arg == "-ftpwSigmaAlbedo") || (arg == "--filterTPwSigmaAlbedo"))
																																																																											{
																																																																												if (i + 1 < argc)
																																																																												{
																																																																													kernelOpt.tPwSigmaAlbedo = std::atof(argv[++i]);
																																																																												}
																																																																												else
																																																																												{
																																																																													std::cerr << "--filterTPwSigmaAlbedo option requires one argument." << std::endl;
																																																																													return 0;
																																																																												}
																																																																											}
																																																																											else
																																																																												if ((arg == "-ftpwSigmaDistance") || (arg == "--filterTPwSigmaDistance"))
																																																																												{
																																																																													if (i + 1 < argc)
																																																																													{
																																																																														kernelOpt.tPwSigmaDistance = std::atof(argv[++i]);
																																																																													}
																																																																													else
																																																																													{
																																																																														std::cerr << "--filterTPwSigmaDistance option requires one argument." << std::endl;
																																																																														return 0;
																																																																													}
																																																																												}
																																																																												else
																																																																													if ((arg == "-ftpwst") || (arg == "--filterTPwSpaceTreshold"))
																																																																													{
																																																																														if (i + 1 < argc)
																																																																														{
																																																																															kernelOpt.tPwSpaceTreshold = std::atof(argv[++i]);
																																																																														}
																																																																														else
																																																																														{
																																																																															std::cerr << "--filterTPwSpaceTreshold option requires one argument." << std::endl;
																																																																															return 0;
																																																																														}
																																																																													}
																																																																													else
																																																																														if ((arg == "-ftpwi") || (arg == "--filterTPwIteration"))
																																																																														{
																																																																															if (i + 1 < argc)
																																																																															{
																																																																																kernelOpt.tPwIterations = std::atoi(argv[++i]);
																																																																															}
																																																																															else
																																																																															{
																																																																																std::cerr << "--filterTPwIteration option requires one argument." << std::endl;
																																																																																return 0;
																																																																															}
																																																																														}
	}
	return true;
}

void writeRgba(const char fileName[],
	const Rgba *pixels,
	int width,
	int height)
{
	RgbaOutputFile file(fileName, width, height, WRITE_RGBA);
	file.setFrameBuffer(pixels, 1, width);
	file.writePixels(height);
}

void readImage(const char fileName[],
	Array2D<Rgba> &pixels,
	int &width,
	int &height)
{
	RgbaInputFile file(fileName);
	Box2i dw = file.dataWindow();

	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	pixels.resizeErase(height, width);

	file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
	file.readPixels(dw.min.y, dw.max.y);
}

typedef struct GZ
{
	half g;
	float z;
};

void readGZ2(const char fileName[],
	Array2D<GZ> &pixels,
	int &width, int &height)
{
	//
	// Read an image using class InputFile.  Try to read one channel,
	// G, of type HALF, and one channel, Z, of type FLOAT.  In memory,
	// the G and Z channels will be interleaved in a single buffer.
	//
	//	- open the file
	//	- allocate memory for the pixels
	//	- describe the layout of the GZ pixel buffer
	//	- read the pixels from the file
	//

	InputFile file(fileName);
	Box2i dw = file.header().dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	int dx = dw.min.x;
	int dy = dw.min.y;

	pixels.resizeErase(height, width);

	FrameBuffer frameBuffer;
	frameBuffer.insert("G",					 // name
		Slice(Imf::HALF,			 // type
		(char *)&pixels[-dy][-dx].g,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	frameBuffer.insert("Z",					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].z,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	file.setFrameBuffer(frameBuffer);
	file.readPixels(dw.min.y, dw.max.y);
}

void writeGZ1(const char fileName[],
	const half *gPixels,
	const float *zPixels,
	int width,
	int height)
{
	//
	// Write an image with only a G (green) and a Z (depth) channel,
	// using class OutputFile.
	//
	//	- create a file header
	//	- add G and Z channels to the header
	//	- open the file, and store the header in the file
	//	- describe the memory layout of the G anx Z pixels
	//	- store the pixels in the file
	//

	Header header(width, height);
	header.channels().insert("light.G", Channel(Imf_2_2::HALF));
	header.channels().insert("light.Z", Channel(Imf_2_2::FLOAT));

	OutputFile file(fileName, header);

	FrameBuffer frameBuffer;

	frameBuffer.insert("light.G",					// name
		Slice(Imf_2_2::HALF,			// type
		(char *)gPixels,		// base
			sizeof(*gPixels) * 1,		// xStride
			sizeof(*gPixels) * width));	// yStride

	frameBuffer.insert("light.Z",					// name
		Slice(Imf_2_2::FLOAT,			// type
		(char *)zPixels,		// base
			sizeof(*zPixels) * 1,		// xStride
			sizeof(*zPixels) * width));	// yStride

	file.setFrameBuffer(frameBuffer);
	file.writePixels(height);
}

void readImageSub32bit(const char fileName[], string channelName,
	Array2D<RgbaF> &pixels,
	int &width,
	int &height)
{
	InputFile file(fileName);
	Box2i dw = file.header().dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	int dx = dw.min.x;
	int dy = dw.min.y;

	pixels.resizeErase(height, width);
	string xR = channelName + "r";
	string xG = channelName + "g";
	string xB = channelName + "b";
	string xA = channelName + "a";
	FrameBuffer frameBuffer;
	frameBuffer.insert(xR,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].r,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	frameBuffer.insert(xG,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].g,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	frameBuffer.insert(xB,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].b,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	frameBuffer.insert(xA,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].a,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	file.setFrameBuffer(frameBuffer);
	file.readPixels(dw.min.y, dw.max.y);
}

void readImage32bit(const char fileName[], string channelName,
	Array2D<RgbaF> &pixels,
	int &width,
	int &height)
{
	InputFile file(fileName);
	Box2i dw = file.header().dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	int dx = dw.min.x;
	int dy = dw.min.y;
	pixels.resizeErase(height, width);
	string xR = channelName + "R";
	string xG = channelName + "G";
	string xB = channelName + "B";
	string xA = channelName + "A";
	FrameBuffer frameBuffer;
	frameBuffer.insert(xR,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].r,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	frameBuffer.insert(xG,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].g,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	frameBuffer.insert(xB,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].b,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	frameBuffer.insert(xA,					 // name
		Slice(Imf::FLOAT,			 // type
		(char *)&pixels[-dy][-dx].a,	 // base
			sizeof(pixels[0][0]) * 1,	 // xStride
			sizeof(pixels[0][0]) * width)); // yStride

	file.setFrameBuffer(frameBuffer);
	file.readPixels(dw.min.y, dw.max.y);
}

int getPadding(char *name)
{
	int padding = 0;
	std::string fname = name;
	for (int i = 0; i< fname.length(); i++) {
		if (fname[i] == '#')
			padding++;
	}
	return padding;
}

std::string FillZero(int number, int padding)
{
	std::string snumber;
	std::stringstream out;
	out << number;
	snumber = out.str();
	std::string zerofill = "";
	if ((snumber.length() < padding) && (number > -1))
		for (int i = 0; i < padding - snumber.length(); i++)
			zerofill += "0";
	return zerofill + snumber;
}

std::string FillPads(std::string name, int frame, int padding)
{
	std::string out = name;
	int strIndex = 0;
	for (int i = 0; i < out.length(); i++)
		if (out[i] == '#')
		{
			strIndex = i;
			break;
		}
	for (int i = 0; i < padding; i++)
		out[strIndex + i] = FillZero(frame, padding)[i];
	return out;
}

std::string OutFileName(std::string SourceName, int frame, int padding, char *oPostfix)
{
	std::string out = SourceName;
	int strIndex = 0;
	for (int i = 0; i < out.length(); i++)
		if (out[i] == '#')
		{
			strIndex = i;
			break;
		}
	for (int i = 0; i < padding; i++)
		out[strIndex + i] = FillZero(frame, padding)[i];
	std::string pre;
	for (int i = 0; i < strIndex - 1; i++)
		pre += out[i];
	pre += oPostfix;
	for (int i = strIndex - 1; i < out.length(); i++)
		pre += out[i];
	return pre;
}

void readChannel(int frameNumber,
	char *chnfName,
	Array<Array2D<Rgba>> &pixels,
	imageOptions &iOpt)
{
	// load beauty image sequence
	std::string chnameLen = chnfName;
	if (chnameLen.length() != 0)
		if (getPadding(chnfName) != 0) {
			int indexArrayExists = 0;
			for (int i = frameNumber; i < frameNumber + int((iOpt.nFrames - 1) / 2) + 1; i++)
				if ((i >= iOpt.startFrame) && (i <= iOpt.endFrame)) {
					std::string filePath = chnfName;
					filePath = FillPads(filePath, i, getPadding(chnfName));
					readImage(filePath.c_str(), pixels[indexArrayExists], iOpt.with, iOpt.height);
					iOpt.existsFrame[indexArrayExists] = 1;
					indexArrayExists++;
				}
				else
				{
					iOpt.existsFrame[indexArrayExists] = 0;
					indexArrayExists++;
				}
			for (int i = frameNumber - 1; i > frameNumber - int((iOpt.nFrames - 1) / 2) - 1; i--)
				if ((i >= iOpt.startFrame) && (i <= iOpt.endFrame)) {
					std::string filePath = chnfName;
					filePath = FillPads(filePath, i, getPadding(chnfName));
					readImage(filePath.c_str(), pixels[indexArrayExists], iOpt.with, iOpt.height);
					iOpt.existsFrame[indexArrayExists] = 1;
					indexArrayExists++;
				}
				else
				{
					iOpt.existsFrame[indexArrayExists] = 0;
					indexArrayExists++;
				}
		}
		else
		{
			for (int i = 0; i < iOpt.nFrames; i++)
				iOpt.existsFrame[i] = 0;
			iOpt.existsFrame[0] = 1;
			readImage(chnfName, pixels[0], iOpt.with, iOpt.height);
		}
}

void readChannel32bit(int frameNumber,
	char *chnfName,
	Array<Array2D<RgbaF>> &pixels,
	imageOptions &iOpt, string channelName)
{
	// load beauty image sequence
	std::string chnameLen = chnfName;
	if (chnameLen.length() != 0)
		if (getPadding(chnfName) != 0) {
			int indexArrayExists = 0;
			for (int i = frameNumber; i < frameNumber + int((iOpt.nFrames - 1) / 2) + 1; i++)
				if ((i >= iOpt.startFrame) && (i <= iOpt.endFrame)) {
					std::string filePath = chnfName;
					filePath = FillPads(filePath, i, getPadding(chnfName));
					readImage32bit(filePath.c_str(), channelName, pixels[indexArrayExists], iOpt.with, iOpt.height);
					iOpt.existsFrame[indexArrayExists] = 1;
					indexArrayExists++;
				}
				else
				{
					iOpt.existsFrame[indexArrayExists] = 0;
					indexArrayExists++;
				}
			for (int i = frameNumber - 1; i > frameNumber - int((iOpt.nFrames - 1) / 2) - 1; i--)
				if ((i >= iOpt.startFrame) && (i <= iOpt.endFrame)) {
					std::string filePath = chnfName;
					filePath = FillPads(filePath, i, getPadding(chnfName));
					readImage32bit(filePath.c_str(), channelName, pixels[indexArrayExists], iOpt.with, iOpt.height);
					iOpt.existsFrame[indexArrayExists] = 1;
					indexArrayExists++;
				}
				else
				{
					iOpt.existsFrame[indexArrayExists] = 0;
					indexArrayExists++;
				}
		}
		else
		{
			for (int i = 0; i < iOpt.nFrames; i++)
				iOpt.existsFrame[i] = 0;
			iOpt.existsFrame[0] = 1;
			readImage32bit(chnfName, channelName, pixels[0], iOpt.with, iOpt.height);
		}
}

void readChannelSub32bit(int frameNumber,
	char *chnfName,
	Array<Array2D<RgbaF>> &pixels,
	imageOptions &iOpt, string channelName)
{
	// load beauty image sequence
	std::string chnameLen = chnfName;
	if (chnameLen.length() != 0)
		if (getPadding(chnfName) != 0) {
			int indexArrayExists = 0;
			for (int i = frameNumber; i < frameNumber + int((iOpt.nFrames - 1) / 2) + 1; i++)
				if ((i >= iOpt.startFrame) && (i <= iOpt.endFrame)) {
					std::string filePath = chnfName;
					filePath = FillPads(filePath, i, getPadding(chnfName));
					readImageSub32bit(filePath.c_str(), channelName, pixels[indexArrayExists], iOpt.with, iOpt.height);
					iOpt.existsFrame[indexArrayExists] = 1;
					indexArrayExists++;
				}
				else
				{
					iOpt.existsFrame[indexArrayExists] = 0;
					indexArrayExists++;
				}
			for (int i = frameNumber - 1; i > frameNumber - int((iOpt.nFrames - 1) / 2) - 1; i--)
				if ((i >= iOpt.startFrame) && (i <= iOpt.endFrame)) {
					std::string filePath = chnfName;
					filePath = FillPads(filePath, i, getPadding(chnfName));
					readImageSub32bit(filePath.c_str(), channelName, pixels[indexArrayExists], iOpt.with, iOpt.height);
					iOpt.existsFrame[indexArrayExists] = 1;
					indexArrayExists++;
				}
				else
				{
					iOpt.existsFrame[indexArrayExists] = 0;
					indexArrayExists++;
				}
		}
		else
		{
			for (int i = 0; i < iOpt.nFrames; i++)
				iOpt.existsFrame[i] = 0;
			iOpt.existsFrame[0] = 1;
			readImageSub32bit(chnfName, channelName, pixels[0], iOpt.with, iOpt.height);
		}
}

void readSingleChannel32bit(int frameNumber,
	char *chnfName,
	Array2D<RgbaF> &pixels,
	imageOptions &iOpt, string channelName)
{
	// load beauty image sequence
	std::string chnameLen = chnfName;
	if (chnameLen.length() != 0)
		if (getPadding(chnfName) != 0) {
			if ((frameNumber >= iOpt.startFrame) && (frameNumber <= iOpt.endFrame)) {
				std::string filePath = chnfName;
				filePath = FillPads(filePath, frameNumber, getPadding(chnfName));
				readImage32bit(filePath.c_str(), channelName, pixels, iOpt.with, iOpt.height);
			}
		}
		else
		{
			readImage32bit(chnfName, channelName, pixels, iOpt.with, iOpt.height);
		}
}

void readSingleChannelSub32bit(int frameNumber,
	char *chnfName,
	Array2D<RgbaF> &pixels,
	imageOptions &iOpt, string channelName)
{
	// load beauty image sequence
	std::string chnameLen = chnfName;
	if (chnameLen.length() != 0)
		if (getPadding(chnfName) != 0) {
			if ((frameNumber >= iOpt.startFrame) && (frameNumber <= iOpt.endFrame)) {
				std::string filePath = chnfName;
				filePath = FillPads(filePath, frameNumber, getPadding(chnfName));
				readImageSub32bit(filePath.c_str(), channelName, pixels, iOpt.with, iOpt.height);
			}
		}
		else
		{
			readImageSub32bit(chnfName, channelName, pixels, iOpt.with, iOpt.height);
		}
}

void readFrame(int frameNumber,
	imgPixels &pixels,
	imageOptions &iOpt)
{
	iOpt.existsFrame.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsAlbedo.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsBeauty.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsPosition.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsIndirectDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsIndirectSpecular.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsSpecular.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsRefraction.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsNormal.resizeEraseUnsafe(iOpt.nFrames);
	pixels.pixelsDepth.resizeEraseUnsafe(iOpt.nFrames);

	readChannel32bit(frameNumber, iOpt.fnameBeauty, pixels.pixelsBeauty, iOpt, "");
	for (int i = 0; i < iOpt.nFrames; i++) {
		pixels.pixelsNormal[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsDepth[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsAlbedo[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsPosition[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsIndirectDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsIndirectSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		pixels.pixelsRefraction[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	}
	// initialize
	int wResize = iOpt.with;
	int hResize = iOpt.height;
	int div = 1;
	while (div != 0) {
		div = wResize % kernelOpt.tBlockSize;
		if (div != 0)
			wResize++;
	}
	div = 1;
	while (div != 0) {
		div = hResize % kernelOpt.tBlockSize;
		if (div != 0)
			hResize++;
	}
	blockMV.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		blockMV[i].resizeEraseUnsafe((wResize / kernelOpt.tBlockSize), (hResize / kernelOpt.tBlockSize));
	blockMVPosition.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		blockMVPosition[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothPosition.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothPosition[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	tSWeight.resizeEraseUnsafe(iOpt.with, iOpt.height);
	tDWeight.resizeEraseUnsafe(iOpt.with, iOpt.height);
	FireflyWeight.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_res.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_resT.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_resSpecular.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_resAlbedo.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_resDiffuse.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_resIndDiffuse.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_resIndSpecular.resizeEraseUnsafe(iOpt.with, iOpt.height);
	pix_resRefraction.resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsBeauty.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsBeauty[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsAlbedo.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsAlbedo[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsIndirectDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsIndirectDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsIndirectSpecular.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsIndirectSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsRefraction.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsRefraction[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsSpecular.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	//tech channel
	ResultBlockSmooth.pixelsDepth.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsDepth[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsNormal.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsNormal[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmooth.pixelsPosition.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmooth.pixelsPosition[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsBeauty.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsBeauty[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsAlbedo.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsAlbedo[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsIndirectDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsIndirectDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsIndirectSpecular.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsIndirectSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsRefraction.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsRefraction[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsSpecular.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	//tech channel
	ResultBlockSmoothTemporal.pixelsDepth.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsDepth[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsNormal.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsNormal[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporal.pixelsPosition.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++)
		ResultBlockSmoothTemporal.pixelsPosition[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	/// tmp
	ResultBlockSmoothTemporalTMP.pixelsBeauty.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsBeauty[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsAlbedo.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsAlbedo[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsDiffuse.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsDiffuse[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsIndirectDiffuse.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsIndirectDiffuse[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsIndirectSpecular.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsIndirectSpecular[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsRefraction.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsRefraction[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsSpecular.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsSpecular[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsDepth.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsDepth[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsNormal.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsNormal[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTemporalTMP.pixelsPosition.resizeEraseUnsafe(1);
	ResultBlockSmoothTemporalTMP.pixelsPosition[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTMP.pixelsBeauty.resizeEraseUnsafe(iOpt.nFrames);
	ResultBlockSmoothTMP.pixelsAlbedo.resizeEraseUnsafe(iOpt.nFrames);
	ResultBlockSmoothTMP.pixelsDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	ResultBlockSmoothTMP.pixelsIndirectDiffuse.resizeEraseUnsafe(iOpt.nFrames);
	ResultBlockSmoothTMP.pixelsIndirectSpecular.resizeEraseUnsafe(iOpt.nFrames);
	ResultBlockSmoothTMP.pixelsRefraction.resizeEraseUnsafe(iOpt.nFrames);
	ResultBlockSmoothTMP.pixelsSpecular.resizeEraseUnsafe(iOpt.nFrames);
	for (int i = 0; i < iOpt.nFrames; i++) {
		ResultBlockSmoothTMP.pixelsBeauty[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		ResultBlockSmoothTMP.pixelsAlbedo[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		ResultBlockSmoothTMP.pixelsDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		ResultBlockSmoothTMP.pixelsIndirectDiffuse[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		ResultBlockSmoothTMP.pixelsIndirectSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		ResultBlockSmoothTMP.pixelsRefraction[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
		ResultBlockSmoothTMP.pixelsSpecular[i].resizeEraseUnsafe(iOpt.with, iOpt.height);
	}
	ResultBlockSmoothTMP.pixelsDepth.resizeEraseUnsafe(1);
	ResultBlockSmoothTMP.pixelsDepth[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTMP.pixelsNormal.resizeEraseUnsafe(1);
	ResultBlockSmoothTMP.pixelsNormal[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	ResultBlockSmoothTMP.pixelsPosition.resizeEraseUnsafe(1);
	ResultBlockSmoothTMP.pixelsPosition[0].resizeEraseUnsafe(iOpt.with, iOpt.height);
	readChannel32bit(frameNumber, iOpt.fnameAlbedo, pixels.pixelsAlbedo, iOpt, "");
	readChannel32bit(frameNumber, iOpt.fnameDepth, pixels.pixelsDepth, iOpt, "");
	readChannel32bit(frameNumber, iOpt.fnameNormal, pixels.pixelsNormal, iOpt, "");
	if (iOpt.exrlayers == "rendermanV") {
		readChannelSub32bit(frameNumber, iOpt.fnameDiffuse, pixels.pixelsDiffuse, iOpt, "diffuse0.");
		readChannelSub32bit(frameNumber, iOpt.fnameIndirectDiffuse, pixels.pixelsIndirectDiffuse, iOpt, "indirectdiffuse0.");
		readChannelSub32bit(frameNumber, iOpt.fnameIndirectSpecular, pixels.pixelsIndirectSpecular, iOpt, "indirectspecular0.");
		readChannelSub32bit(frameNumber, iOpt.fnameSpecular, pixels.pixelsSpecular, iOpt, "specular0.");
	}
	if (iOpt.exrlayers == "standard")
	{
		readChannel32bit(frameNumber, iOpt.fnameDiffuse, pixels.pixelsDiffuse, iOpt, "");
		readChannel32bit(frameNumber, iOpt.fnameIndirectDiffuse, pixels.pixelsIndirectDiffuse, iOpt, "");
		readChannel32bit(frameNumber, iOpt.fnameIndirectSpecular, pixels.pixelsIndirectSpecular, iOpt, "");
		readChannel32bit(frameNumber, iOpt.fnameSpecular, pixels.pixelsSpecular, iOpt, "");
	}
	readChannel32bit(frameNumber, iOpt.fnameRefraction, pixels.pixelsRefraction, iOpt, "");
	readChannel32bit(frameNumber, iOpt.fnamePosition, pixels.pixelsPosition, iOpt, "");
}

void BilinearInterpolation(Array<Array2D<Rgba>> &blockMV, Array<Array2D<Rgba>> &resultMV, imageOptions &iOpt, int BlockSize)
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;
	int div = 1;
	while (div != 0) {
		div = wResize % BlockSize;
		if (div != 0)
			wResize++;
	}
	div = 1;
	while (div != 0) {
		div = hResize % BlockSize;
		if (div != 0)
			hResize++;
	}
	Rgba *mvRes = &resultMV[0][0][0];
	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = 1; bx < wResize / BlockSize - 1; bx++)
				for (int by = 1; by < hResize / BlockSize - 1; by++)
					for (int x = 0; x < BlockSize; x++)
						for (int y = 0; y < BlockSize; y++)
						{
							float x2x1, y2y1, x2x, y2y, yy1, xx1;
							Rgba q11, q12, q21, q22;
							x2x1 = BlockSize;
							y2y1 = BlockSize;
							x2x = BlockSize - x;
							y2y = BlockSize - y;
							yy1 = y;
							xx1 = x;
							q11.r = (blockMV[frame][bx][by].r + blockMV[frame][bx - 1][by].r + blockMV[frame][bx - 1][by - 1].r + blockMV[frame][bx][by - 1].r) / 4;
							q11.g = (blockMV[frame][bx][by].g + blockMV[frame][bx - 1][by].g + blockMV[frame][bx - 1][by - 1].g + blockMV[frame][bx][by - 1].g) / 4;

							q12.r = (blockMV[frame][bx][by].r + blockMV[frame][bx - 1][by].r + blockMV[frame][bx - 1][by + 1].r + blockMV[frame][bx][by + 1].r) / 4;
							q12.g = (blockMV[frame][bx][by].g + blockMV[frame][bx - 1][by].g + blockMV[frame][bx - 1][by + 1].g + blockMV[frame][bx][by + 1].g) / 4;

							q21.r = (blockMV[frame][bx][by].r + blockMV[frame][bx + 1][by].r + blockMV[frame][bx + 1][by - 1].r + blockMV[frame][bx][by - 1].r) / 4;
							q21.g = (blockMV[frame][bx][by].g + blockMV[frame][bx + 1][by].g + blockMV[frame][bx + 1][by - 1].g + blockMV[frame][bx][by - 1].g) / 4;

							q22.r = (blockMV[frame][bx][by].r + blockMV[frame][bx + 1][by].r + blockMV[frame][bx + 1][by + 1].r + blockMV[frame][bx][by + 1].r) / 4;
							q22.g = (blockMV[frame][bx][by].g + blockMV[frame][bx + 1][by].g + blockMV[frame][bx + 1][by + 1].g + blockMV[frame][bx][by + 1].g) / 4;

							mvRes = &resultMV[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;
							mvRes->r = (1.0 / (x2x1 * y2y1))* (q11.r * x2x * y2y +
								q21.r * xx1 * y2y +
								q12.r * x2x * yy1 +
								q22.r * xx1 * yy1);
							mvRes->g = (1.0 / (x2x1 * y2y1))* (q11.g * x2x * y2y +
								q21.g * xx1 * y2y +
								q12.g * x2x * yy1 +
								q22.g * xx1 * yy1);
							mvRes->b = 0;
							mvRes->a = 1;
						}
}

void BlockSmooth(int core)
{
	int BlockSize = kernelOpt.tBlockSize;
	float Size = kernelOpt.tInterpolation;
	int wResize, hResize;
	int w = iOpt.with;
	int h = iOpt.height;

	wResize = iOpt.with;
	hResize = iOpt.height;
	int div = 1;
	while (div != 0) {
		div = wResize % BlockSize;
		if (div != 0)
			wResize++;
	}

	div = 1;
	while (div != 0) {
		div = hResize % BlockSize;
		if (div != 0)
			hResize++;
	}

	RgbaF *pixBeauty;
	RgbaF *ResultBPix;

	RgbaF *pixAlbedo;
	RgbaF *ResultAlbedoPix;

	RgbaF *pixSpec;
	RgbaF *ResultSpecPix;

	RgbaF *pixDiffuse;
	RgbaF *ResultDiffusePix;

	RgbaF *pixIndDiffuse;
	RgbaF *ResultIndDiffusePix;

	RgbaF *pixIndSpec;
	RgbaF *ResultIndSpecPix;

	RgbaF *pixRefract;
	RgbaF *ResultRefractPix;

	//tech channel
	RgbaF *pixDepth;
	RgbaF *ResultDepthPix;

	RgbaF *pixNormal;
	RgbaF *ResultNormalPix;

	RgbaF *pixPosition;
	RgbaF *ResultPositionPix;

	Rgba *tSWeightPix;
	Rgba *tDWeightPix;

	int frameCount = iOpt.nFrames;
	int CoreSize = int(frameCount / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int frame = 0; frame < 1; frame++)
		if (frame < iOpt.nFrames)
			if (iOpt.existsFrame[frame] == 1)
				for (int bx = 0; bx < w; bx++)
					for (int by = 0; by < h; by++) {

						tSWeightPix = &tSWeight[0][0] + bx + by*iOpt.with;
						tDWeightPix = &tDWeight[0][0] + bx + by*iOpt.with;

						pixBeauty = &pixels.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						ResultBPix = &ResultBlockSmoothTemporal.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;

						pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
						ResultAlbedoPix = &ResultBlockSmoothTemporal.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;

						pixSpec = &pixels.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;
						ResultSpecPix = &ResultBlockSmoothTemporal.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

						pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						ResultDiffusePix = &ResultBlockSmoothTemporal.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;

						pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						ResultIndDiffusePix = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;

						pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						ResultIndSpecPix = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;

						pixRefract = &pixels.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						ResultRefractPix = &ResultBlockSmoothTemporal.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;

						//tech channel
						pixDepth = &pixels.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
						ResultDepthPix = &ResultBlockSmoothTemporal.pixelsDepth[frame][0][0] + bx + by*iOpt.with;

						pixNormal = &pixels.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
						ResultNormalPix = &ResultBlockSmoothTemporal.pixelsNormal[frame][0][0] + bx + by*iOpt.with;

						pixPosition = &pixels.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
						ResultPositionPix = &ResultBlockSmoothTemporal.pixelsPosition[frame][0][0] + bx + by*iOpt.with;

						ResultBPix->a = pixBeauty->a;
						ResultBPix->r = pixBeauty->r;
						ResultBPix->g = pixBeauty->g;
						ResultBPix->b = pixBeauty->b;

						ResultAlbedoPix->r = pixAlbedo->r;
						ResultAlbedoPix->g = pixAlbedo->g;
						ResultAlbedoPix->b = pixAlbedo->b;

						ResultSpecPix->r = pixSpec->r;
						ResultSpecPix->g = pixSpec->g;
						ResultSpecPix->b = pixSpec->b;

						ResultDiffusePix->r = pixDiffuse->r;
						ResultDiffusePix->g = pixDiffuse->g;
						ResultDiffusePix->b = pixDiffuse->b;

						ResultIndDiffusePix->r = pixIndDiffuse->r;
						ResultIndDiffusePix->g = pixIndDiffuse->g;
						ResultIndDiffusePix->b = pixIndDiffuse->b;

						ResultIndSpecPix->r = pixIndSpec->r;
						ResultIndSpecPix->g = pixIndSpec->g;
						ResultIndSpecPix->b = pixIndSpec->b;

						ResultRefractPix->r = pixRefract->r;
						ResultRefractPix->g = pixRefract->g;
						ResultRefractPix->b = pixRefract->b;

						//tech channel
						ResultDepthPix->r = pixDepth->r;
						ResultDepthPix->g = pixDepth->g;
						ResultDepthPix->b = pixDepth->b;

						ResultNormalPix->r = pixNormal->r;
						ResultNormalPix->g = pixNormal->g;
						ResultNormalPix->b = pixNormal->b;

						ResultPositionPix->r = pixPosition->r;
						ResultPositionPix->g = pixPosition->g;
						ResultPositionPix->b = pixPosition->b;
					}
	frameCount = iOpt.nFrames;
	CoreSize = int(frameCount / iOpt.nCores) + 1;
	startX = CoreSize*core;
	endX = CoreSize*(core + 1);
	for (int frame = startX; frame < endX; frame++)
		if ((frame > 0) && (frame < iOpt.nFrames))
			if (iOpt.existsFrame[frame] == 1)
				for (int bx = 0; bx < wResize / BlockSize; bx++)
					for (int by = 0; by < hResize / BlockSize; by++)
					{
						int mvg, mvr;
						float fmvr = float(blockMV[frame][bx][by].r);
						float fmvg = float(blockMV[frame][bx][by].g);
						if (fmvr > 0) {
							mvr = (int)(fmvr + 0.5);
						}
						else
						{
							mvr = (int)(fmvr - 0.5);
						}
						if (fmvg > 0) {
							mvg = (int)(fmvg + 0.5);
						}
						else
						{
							mvg = (int)(fmvg - 0.5);
						}
						for (int x = -Size*BlockSize; x < BlockSize + Size*BlockSize; x++)
							for (int y = -Size*BlockSize; y < BlockSize + Size*BlockSize; y++)
								if (((bx*BlockSize + x) >= 0) && ((bx*BlockSize + x)<w))
									if (((by*BlockSize + y) >= 0) && ((by*BlockSize + y)<h))
										if (((bx*BlockSize + mvr + x) >= 0) && ((bx*BlockSize + mvr + x)<w))
											if (((by*BlockSize + mvg + y) >= 0) && ((by*BlockSize + mvg + y)<h))
											{
												float weight = 1;

												tSWeightPix = &tSWeight[0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;
												tDWeightPix = &tDWeight[0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												pixBeauty = &pixels.pixelsBeauty[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultBPix = &ResultBlockSmoothTemporal.pixelsBeauty[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultAlbedoPix = &ResultBlockSmoothTemporal.pixelsAlbedo[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												pixSpec = &pixels.pixelsSpecular[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultSpecPix = &ResultBlockSmoothTemporal.pixelsSpecular[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;


												pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultDiffusePix = &ResultBlockSmoothTemporal.pixelsDiffuse[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;


												pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultIndDiffusePix = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultIndSpecPix = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												pixRefract = &pixels.pixelsRefraction[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultRefractPix = &ResultBlockSmoothTemporal.pixelsRefraction[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												// tech
												pixDepth = &pixels.pixelsDepth[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultDepthPix = &ResultBlockSmoothTemporal.pixelsDepth[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												pixNormal = &pixels.pixelsNormal[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultNormalPix = &ResultBlockSmoothTemporal.pixelsNormal[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												pixPosition = &pixels.pixelsPosition[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + mvr + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + mvg + y, 0), iOpt.height - 1)*iOpt.with;
												ResultPositionPix = &ResultBlockSmoothTemporal.pixelsPosition[frame][0][0] + std::min(std::max(std::max(bx*BlockSize, 0) + x, 0), iOpt.with - 1) + std::min(std::max(std::max(by*BlockSize, 0) + y, 0), iOpt.height - 1)*iOpt.with;

												if (x <= 0) {
													weight = (x / (Size*BlockSize*1.0)) + 1;
												}
												if (x >= BlockSize) {
													weight = 1 - ((x - BlockSize) / (Size*BlockSize*1.0));
												}
												if ((x>0) && (x<BlockSize)) {
													weight = 1;
												}

												if (y <= 0) {
													weight *= (y / (Size*BlockSize*1.0)) + 1;
												}
												if (y >= BlockSize) {
													weight *= 1 - ((y - BlockSize) / (Size*BlockSize*1.0));
												}
												if ((y>0) && (y<BlockSize)) {
													weight *= 1;
												}
												if (weight < 0)
													weight = 0;
												if (weight > 1)
													weight = 1;
												ResultBPix->a = ResultBPix->a*(1 - weight*float(blockMV[frame][bx][by].a)) + pixBeauty->a*weight*float(blockMV[frame][bx][by].a);
												ResultBPix->r = ResultBPix->r*(1 - weight) + pixBeauty->r*weight;
												ResultBPix->g = ResultBPix->g*(1 - weight) + pixBeauty->g*weight;
												ResultBPix->b = ResultBPix->b*(1 - weight) + pixBeauty->b*weight;

												ResultAlbedoPix->a = 1;
												ResultAlbedoPix->r = ResultAlbedoPix->r*(1 - weight) + pixAlbedo->r*weight;
												ResultAlbedoPix->g = ResultAlbedoPix->g*(1 - weight) + pixAlbedo->g*weight;
												ResultAlbedoPix->b = ResultAlbedoPix->b*(1 - weight) + pixAlbedo->b*weight;

												ResultSpecPix->a = 1;
												ResultSpecPix->r = ResultSpecPix->r*(1 - weight) + pixSpec->r*weight;
												ResultSpecPix->g = ResultSpecPix->g*(1 - weight) + pixSpec->g*weight;
												ResultSpecPix->b = ResultSpecPix->b*(1 - weight) + pixSpec->b*weight;

												ResultDiffusePix->a = 1;
												ResultDiffusePix->r = ResultDiffusePix->r*(1 - weight) + pixDiffuse->r*weight;
												ResultDiffusePix->g = ResultDiffusePix->g*(1 - weight) + pixDiffuse->g*weight;
												ResultDiffusePix->b = ResultDiffusePix->b*(1 - weight) + pixDiffuse->b*weight;

												ResultIndDiffusePix->a = 1;
												ResultIndDiffusePix->r = ResultIndDiffusePix->r*(1 - weight) + pixIndDiffuse->r*weight;
												ResultIndDiffusePix->g = ResultIndDiffusePix->g*(1 - weight) + pixIndDiffuse->g*weight;
												ResultIndDiffusePix->b = ResultIndDiffusePix->b*(1 - weight) + pixIndDiffuse->b*weight;

												ResultIndSpecPix->a = 1;
												ResultIndSpecPix->r = ResultIndSpecPix->r*(1 - weight) + pixIndSpec->r*weight;
												ResultIndSpecPix->g = ResultIndSpecPix->g*(1 - weight) + pixIndSpec->g*weight;
												ResultIndSpecPix->b = ResultIndSpecPix->b*(1 - weight) + pixIndSpec->b*weight;

												ResultRefractPix->a = 1;
												ResultRefractPix->r = ResultRefractPix->r*(1 - weight) + pixRefract->r*weight;
												ResultRefractPix->g = ResultRefractPix->g*(1 - weight) + pixRefract->g*weight;
												ResultRefractPix->b = ResultRefractPix->b*(1 - weight) + pixRefract->b*weight;

												ResultRefractPix->a = 1;
												ResultRefractPix->r = ResultRefractPix->r*(1 - weight) + pixRefract->r*weight;
												ResultRefractPix->g = ResultRefractPix->g*(1 - weight) + pixRefract->g*weight;
												ResultRefractPix->b = ResultRefractPix->b*(1 - weight) + pixRefract->b*weight;

												// tech
												ResultDepthPix->a = 1;
												ResultDepthPix->r = ResultDepthPix->r*(1 - weight) + pixDepth->r*weight;
												ResultDepthPix->g = ResultDepthPix->g*(1 - weight) + pixDepth->g*weight;
												ResultDepthPix->b = ResultDepthPix->b*(1 - weight) + pixDepth->b*weight;

												ResultNormalPix->a = 1;
												ResultNormalPix->r = ResultNormalPix->r*(1 - weight) + pixNormal->r*weight;
												ResultNormalPix->g = ResultNormalPix->g*(1 - weight) + pixNormal->g*weight;
												ResultNormalPix->b = ResultNormalPix->b*(1 - weight) + pixNormal->b*weight;

												ResultPositionPix->a = 1;
												ResultPositionPix->r = ResultPositionPix->r*(1 - weight) + pixPosition->r*weight;
												ResultPositionPix->g = ResultPositionPix->g*(1 - weight) + pixPosition->g*weight;
												ResultPositionPix->b = ResultPositionPix->b*(1 - weight) + pixPosition->b*weight;
											}
					}
}

void unpremult(int core)
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;

	RgbaF *pixBeauty;
	RgbaF *ResultBPix;

	RgbaF *pixAlbedo;
	RgbaF *ResultAlbedoPix;

	RgbaF *pixSpec;
	RgbaF *ResultSpecPix;

	RgbaF *pixDiffuse;
	RgbaF *ResultDiffusePix;

	RgbaF *pixIndDiffuse;
	RgbaF *ResultIndDiffusePix;

	RgbaF *pixIndSpec;
	RgbaF *ResultIndSpecPix;

	RgbaF *pixRefract;
	RgbaF *ResultRefractPix;

	int CoreSize = int(wResize / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int frame = 0; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = startX; bx < endX; bx++)
				for (int by = 0; by < hResize; by++)
					if (bx < wResize) {
						pixBeauty = &pixels.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						pixSpec = &pixels.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;
						pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						pixRefract = &pixels.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						if (pixBeauty->a != 0) {
							pixBeauty->r /= pixBeauty->a;
							pixBeauty->g /= pixBeauty->a;
							pixBeauty->b /= pixBeauty->a;

							pixSpec->r /= pixBeauty->a;
							pixSpec->g /= pixBeauty->a;
							pixSpec->b /= pixBeauty->a;

							pixDiffuse->r /= pixBeauty->a;
							pixDiffuse->g /= pixBeauty->a;
							pixDiffuse->b /= pixBeauty->a;

							pixIndDiffuse->r /= pixBeauty->a;
							pixIndDiffuse->g /= pixBeauty->a;
							pixIndDiffuse->b /= pixBeauty->a;

							pixIndSpec->r /= pixBeauty->a;
							pixIndSpec->g /= pixBeauty->a;
							pixIndSpec->b /= pixBeauty->a;

							pixRefract->r /= pixBeauty->a;
							pixRefract->g /= pixBeauty->a;
							pixRefract->b /= pixBeauty->a;
						}
						else
						{
							pixBeauty->r = 0;
							pixBeauty->g = 0;
							pixBeauty->b = 0;

							pixSpec->r = 0;
							pixSpec->g = 0;
							pixSpec->b = 0;

							pixDiffuse->r = 0;
							pixDiffuse->g = 0;
							pixDiffuse->b = 0;

							pixIndDiffuse->r = 0;
							pixIndDiffuse->g = 0;
							pixIndDiffuse->b = 0;

							pixIndSpec->r = 0;
							pixIndSpec->g = 0;
							pixIndSpec->b = 0;

							pixRefract->r = 0;
							pixRefract->g = 0;
							pixRefract->b = 0;
						}
					}
}

void BlockSmoothSpatial(int core)
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;

	RgbaF *pixBeauty;
	RgbaF *ResultBPix;

	RgbaF *pixAlbedo;
	RgbaF *ResultAlbedoPix;

	RgbaF *pixSpec;
	RgbaF *ResultSpecPix;

	RgbaF *pixDiffuse;
	RgbaF *ResultDiffusePix;

	RgbaF *pixIndDiffuse;
	RgbaF *ResultIndDiffusePix;

	RgbaF *pixIndSpec;
	RgbaF *ResultIndSpecPix;

	RgbaF *pixRefract;
	RgbaF *ResultRefractPix;

	//tech channel
	RgbaF *pixDepth;
	RgbaF *ResultDepthPix;

	RgbaF *pixNormal;
	RgbaF *ResultNormalPix;

	RgbaF *pixPosition;
	RgbaF *ResultPositionPix;

	Rgba *tSWeightPix;
	Rgba *tDWeightPix;

	int CoreSize = int(wResize / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int frame = 0; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = startX; bx < endX; bx++)
				for (int by = 0; by < hResize; by++)
					if (bx < wResize) {

						tSWeightPix = &tSWeight[0][0] + bx + by*iOpt.with;
						tDWeightPix = &tDWeight[0][0] + bx + by*iOpt.with;

						pixBeauty = &pixels.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						ResultBPix = &ResultBlockSmooth.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;

						pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
						ResultAlbedoPix = &ResultBlockSmooth.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;

						pixSpec = &pixels.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;
						ResultSpecPix = &ResultBlockSmooth.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;


						pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						ResultDiffusePix = &ResultBlockSmooth.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;

						pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						ResultIndDiffusePix = &ResultBlockSmooth.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;

						pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						ResultIndSpecPix = &ResultBlockSmooth.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;

						pixRefract = &pixels.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						ResultRefractPix = &ResultBlockSmooth.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;

						//tech channel
						pixDepth = &pixels.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
						ResultDepthPix = &ResultBlockSmooth.pixelsDepth[frame][0][0] + bx + by*iOpt.with;

						pixNormal = &pixels.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
						ResultNormalPix = &ResultBlockSmooth.pixelsNormal[frame][0][0] + bx + by*iOpt.with;

						pixPosition = &pixels.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
						ResultPositionPix = &ResultBlockSmooth.pixelsPosition[frame][0][0] + bx + by*iOpt.with;

						ResultBPix->a = pixBeauty->a;
						ResultBPix->r = pixBeauty->r;
						ResultBPix->g = pixBeauty->g;
						ResultBPix->b = pixBeauty->b;

						ResultAlbedoPix->r = pixAlbedo->r;
						ResultAlbedoPix->g = pixAlbedo->g;
						ResultAlbedoPix->b = pixAlbedo->b;

						ResultSpecPix->r = pixSpec->r;
						ResultSpecPix->g = pixSpec->g;
						ResultSpecPix->b = pixSpec->b;

						ResultDiffusePix->r = pixDiffuse->r;
						ResultDiffusePix->g = pixDiffuse->g;
						ResultDiffusePix->b = pixDiffuse->b;

						ResultIndDiffusePix->r = pixIndDiffuse->r;
						ResultIndDiffusePix->g = pixIndDiffuse->g;
						ResultIndDiffusePix->b = pixIndDiffuse->b;

						ResultIndSpecPix->r = pixIndSpec->r;
						ResultIndSpecPix->g = pixIndSpec->g;
						ResultIndSpecPix->b = pixIndSpec->b;

						ResultRefractPix->r = pixRefract->r;
						ResultRefractPix->g = pixRefract->g;
						ResultRefractPix->b = pixRefract->b;

						//tech channel
						ResultDepthPix->r = pixDepth->r;
						ResultDepthPix->g = pixDepth->g;
						ResultDepthPix->b = pixDepth->b;

						ResultNormalPix->r = pixNormal->r;
						ResultNormalPix->g = pixNormal->g;
						ResultNormalPix->b = pixNormal->b;

						ResultPositionPix->r = pixPosition->r;
						ResultPositionPix->g = pixPosition->g;
						ResultPositionPix->b = pixPosition->b;

						tSWeightPix->r = 0;
						tSWeightPix->g = 0;
						tSWeightPix->b = 0;
						tSWeightPix->a = 0;

						tDWeightPix->r = 0;
						tDWeightPix->g = 0;
						tDWeightPix->b = 0;
						tDWeightPix->a = 0;
					}
}

void BlockSmoothSpatial_noCore()
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;

	RgbaF *pixBeauty;
	RgbaF *ResultBPix;

	RgbaF *pixAlbedo;
	RgbaF *ResultAlbedoPix;

	RgbaF *pixSpec;
	RgbaF *ResultSpecPix;

	RgbaF *pixDiffuse;
	RgbaF *ResultDiffusePix;

	RgbaF *pixIndDiffuse;
	RgbaF *ResultIndDiffusePix;

	RgbaF *pixIndSpec;
	RgbaF *ResultIndSpecPix;

	RgbaF *pixRefract;
	RgbaF *ResultRefractPix;

	//tech channel
	RgbaF *pixDepth;
	RgbaF *ResultDepthPix;

	RgbaF *pixNormal;
	RgbaF *ResultNormalPix;

	RgbaF *pixPosition;
	RgbaF *ResultPositionPix;

	Rgba *tSWeightPix;
	Rgba *tDWeightPix;

	for (int frame = 0; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = 0; bx < wResize; bx++)
				for (int by = 0; by < hResize; by++) {

					tSWeightPix = &tSWeight[0][0] + bx + by*iOpt.with;
					tDWeightPix = &tDWeight[0][0] + bx + by*iOpt.with;

					pixBeauty = &pixels.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
					ResultBPix = &ResultBlockSmooth.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;

					pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
					ResultAlbedoPix = &ResultBlockSmooth.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;

					pixSpec = &pixels.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;
					ResultSpecPix = &ResultBlockSmooth.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

					pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
					ResultDiffusePix = &ResultBlockSmooth.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;

					pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
					ResultIndDiffusePix = &ResultBlockSmooth.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;

					pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
					ResultIndSpecPix = &ResultBlockSmooth.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;

					pixRefract = &pixels.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
					ResultRefractPix = &ResultBlockSmooth.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;

					//tech channel
					pixDepth = &pixels.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
					ResultDepthPix = &ResultBlockSmooth.pixelsDepth[frame][0][0] + bx + by*iOpt.with;

					pixNormal = &pixels.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
					ResultNormalPix = &ResultBlockSmooth.pixelsNormal[frame][0][0] + bx + by*iOpt.with;

					pixPosition = &pixels.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
					ResultPositionPix = &ResultBlockSmooth.pixelsPosition[frame][0][0] + bx + by*iOpt.with;

					ResultBPix->a = pixBeauty->a;
					ResultBPix->r = pixBeauty->r;
					ResultBPix->g = pixBeauty->g;
					ResultBPix->b = pixBeauty->b;

					ResultAlbedoPix->r = pixAlbedo->r;
					ResultAlbedoPix->g = pixAlbedo->g;
					ResultAlbedoPix->b = pixAlbedo->b;

					ResultSpecPix->r = pixSpec->r;
					ResultSpecPix->g = pixSpec->g;
					ResultSpecPix->b = pixSpec->b;

					ResultDiffusePix->r = pixDiffuse->r;
					ResultDiffusePix->g = pixDiffuse->g;
					ResultDiffusePix->b = pixDiffuse->b;

					ResultIndDiffusePix->r = pixIndDiffuse->r;
					ResultIndDiffusePix->g = pixIndDiffuse->g;
					ResultIndDiffusePix->b = pixIndDiffuse->b;

					ResultIndSpecPix->r = pixIndSpec->r;
					ResultIndSpecPix->g = pixIndSpec->g;
					ResultIndSpecPix->b = pixIndSpec->b;

					ResultRefractPix->r = pixRefract->r;
					ResultRefractPix->g = pixRefract->g;
					ResultRefractPix->b = pixRefract->b;

					//tech channel
					ResultDepthPix->r = pixDepth->r;
					ResultDepthPix->g = pixDepth->g;
					ResultDepthPix->b = pixDepth->b;

					ResultNormalPix->r = pixNormal->r;
					ResultNormalPix->g = pixNormal->g;
					ResultNormalPix->b = pixNormal->b;

					ResultPositionPix->r = pixPosition->r;
					ResultPositionPix->g = pixPosition->g;
					ResultPositionPix->b = pixPosition->b;
				}
}

void BlockMVSmooth()
{
	int wResize, hResize;
	int BlockSize = kernelOpt.tBlockSize;
	float Threshold = kernelOpt.tMotionTreshold;
	wResize = iOpt.with;
	hResize = iOpt.height;
	int div = 1;
	while (div != 0) {
		div = wResize % BlockSize;
		if (div != 0)
			wResize++;
	}
	div = 1;
	while (div != 0) {
		div = hResize % BlockSize;
		if (div != 0)
			hResize++;
	}

	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = 0; bx < wResize / BlockSize; bx++)
				for (int by = 0; by < hResize / BlockSize; by++) {
					float mvNeiboors[8][3];
					float thisMV[2];
					float deviation = 0;
					thisMV[0] = float(blockMV[frame][bx][by].r);
					thisMV[1] = float(blockMV[frame][bx][by].g);

					if (((bx >= 0) && (bx < wResize / BlockSize)) &&
						((by - 1 >= 0) && (by - 1 < hResize / BlockSize))) {
						mvNeiboors[0][0] = float(blockMV[frame][bx][by - 1].r);
						mvNeiboors[0][1] = float(blockMV[frame][bx][by - 1].g);
						mvNeiboors[0][2] = 1;
					}
					else
					{
						mvNeiboors[0][2] = 0;
					}

					if (((bx + 1 >= 0) && (bx + 1 < wResize / BlockSize)) &&
						((by - 1 >= 0) && (by - 1 < hResize / BlockSize))) {
						mvNeiboors[1][0] = float(blockMV[frame][bx + 1][by - 1].r);
						mvNeiboors[1][1] = float(blockMV[frame][bx + 1][by - 1].g);
						mvNeiboors[1][2] = 1;
					}
					else
					{
						mvNeiboors[1][2] = 0;
					}

					if (((bx + 1 >= 0) && (bx + 1 < wResize / BlockSize)) &&
						((by >= 0) && (by < hResize / BlockSize))) {
						mvNeiboors[2][0] = float(blockMV[frame][bx + 1][by].r);
						mvNeiboors[2][1] = float(blockMV[frame][bx + 1][by].g);
						mvNeiboors[2][2] = 1;
					}
					else
					{
						mvNeiboors[2][2] = 0;
					}

					if (((bx + 1 >= 0) && (bx + 1 < wResize / BlockSize)) &&
						((by + 1 >= 0) && (by + 1 < hResize / BlockSize))) {
						mvNeiboors[3][0] = float(blockMV[frame][bx + 1][by + 1].r);
						mvNeiboors[3][1] = float(blockMV[frame][bx + 1][by + 1].g);
						mvNeiboors[3][2] = 1;
					}
					else
					{
						mvNeiboors[3][2] = 0;
					}

					if (((bx >= 0) && (bx < wResize / BlockSize)) &&
						((by + 1 >= 0) && (by + 1 < hResize / BlockSize))) {
						mvNeiboors[4][0] = float(blockMV[frame][bx][by + 1].r);
						mvNeiboors[4][1] = float(blockMV[frame][bx][by + 1].g);
						mvNeiboors[4][2] = 1;
					}
					else
					{
						mvNeiboors[4][2] = 0;
					}

					if (((bx - 1 >= 0) && (bx - 1 < wResize / BlockSize)) &&
						((by + 1 >= 0) && (by + 1 < hResize / BlockSize))) {
						mvNeiboors[5][0] = float(blockMV[frame][bx - 1][by + 1].r);
						mvNeiboors[5][1] = float(blockMV[frame][bx - 1][by + 1].g);
						mvNeiboors[5][2] = 1;
					}
					else
					{
						mvNeiboors[5][2] = 0;
					}

					if (((bx - 1 >= 0) && (bx - 1 < wResize / BlockSize)) &&
						((by >= 0) && (by < hResize / BlockSize))) {
						mvNeiboors[6][0] = float(blockMV[frame][bx - 1][by].r);
						mvNeiboors[6][1] = float(blockMV[frame][bx - 1][by].g);
						mvNeiboors[6][2] = 1;
					}
					else
					{
						mvNeiboors[6][2] = 0;
					}
					if (((bx - 1 >= 0) && (bx - 1 < wResize / BlockSize)) &&
						((by - 1 >= 0) && (by - 1 < hResize / BlockSize))) {
						mvNeiboors[7][0] = float(blockMV[frame][bx - 1][by - 1].r);
						mvNeiboors[7][1] = float(blockMV[frame][bx - 1][by - 1].g);
						mvNeiboors[7][2] = 1;
					}
					else
					{
						mvNeiboors[7][2] = 0;
					}

					int count = 0;
					for (int i = 0; i < 8; i++)
						if (mvNeiboors[i][2] > 0) {
							deviation += abs((mvNeiboors[i][0] - thisMV[0]) + (mvNeiboors[i][1] - thisMV[1]));
							count++;
						}

					deviation /= count;

					if (deviation > Threshold) {
						blockMV[frame][bx][by].r = 0;
						blockMV[frame][bx][by].g = 0;
						blockMV[frame][bx][by].b = 0;
						blockMV[frame][bx][by].a = 0;
					}
				}
}

void searchBlockDiamond4Step(int core)
{
	int wResize, hResize;
	int BlockSize = kernelOpt.tBlockSize;
	float InterpolationSize = kernelOpt.tInterpolation;
	float Treshold = kernelOpt.tColorTreshold;
	wResize = iOpt.with;
	hResize = iOpt.height;
	int w = iOpt.with;
	int h = iOpt.height;
	int div = 1;
	int Pcount = 1;
	float dcount = 1;

	while (div != 0) {
		div = wResize % BlockSize;
		if (div != 0)
			wResize++;
	}
	div = 1;
	while (div != 0) {
		div = hResize % BlockSize;
		if (div != 0)
			hResize++;
	}

	int nframes = 0;
	for (int i = 1; i < iOpt.nFrames; i++)
		if (iOpt.existsFrame[i] == 1)
			nframes++;

	int CoreSize = int((wResize / BlockSize) / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int xBlock = startX; xBlock < endX; xBlock++)
				for (int yBlock = 0; yBlock < hResize / BlockSize; yBlock++)
					if (xBlock < wResize / BlockSize)
					{
						int coords[13][3];
						int centerX = xBlock*BlockSize;
						int centerY = yBlock*BlockSize;
						int resultCoord[2];
						resultCoord[0] = 1000;
						resultCoord[1] = 1000;
						blockMV[frame][xBlock][yBlock].r = 0;
						blockMV[frame][xBlock][yBlock].g = 0;
						blockMV[frame][xBlock][yBlock].b = 0;
						blockMV[frame][xBlock][yBlock].a = 0;
						float MinSAD = 1000000;
						bool isSearch = false;
						float alpha = 0;
						for (int xb = xBlock*BlockSize; xb < std::min(xBlock*BlockSize + BlockSize, w - 1); xb++)
							for (int yb = yBlock*BlockSize; yb < std::min(yBlock*BlockSize + BlockSize, h - 1); yb++) {
								RgbaF *thisPt = &pixels.pixelsBeauty[frame][0][0] + xb + yb*w;
								alpha += thisPt->a;
							}
						if (alpha > 0)
							for (int iteration = 0; iteration < 4; iteration++) {
								coords[0][0] = centerX;
								coords[0][1] = centerY;
								coords[0][2] = 1;
								int step = (4 - iteration) * 1;
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY - 2 * step >= 0) && (centerY - 2 * step < h)))
								{
									coords[1][0] = centerX;
									coords[1][1] = centerY - 2 * step;
									coords[1][2] = 1;
								}
								else
								{
									coords[1][2] = 0;
								}
								if (((centerX + 2 * step >= 0) && (centerX + 2 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[2][0] = centerX + 2 * step;
									coords[2][1] = centerY;
									coords[2][2] = 1;
								}
								else
								{
									coords[2][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY + 2 * step >= 0) && (centerY + 2 * step < h)))
								{
									coords[3][0] = centerX;
									coords[3][1] = centerY + 2 * step;
									coords[3][2] = 1;
								}
								else
								{
									coords[3][2] = 0;
								}
								if (((centerX - 2 * step >= 0) && (centerX - 2 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[4][0] = centerX - 2 * step;
									coords[4][1] = centerY;
									coords[4][2] = 1;
								}
								else
								{
									coords[4][2] = 0;
								}
								if (((centerX - 1 >= 0) && (centerX - 1 * step < w)) &&
									((centerY - 1 >= 0) && (centerY - 1 * step < h)))
								{
									coords[5][0] = centerX - 1 * step;
									coords[5][1] = centerY - 1 * step;
									coords[5][2] = 1;
								}
								else
								{
									coords[5][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY - 1 * step >= 0) && (centerY - 1 * step < h)))
								{
									coords[6][0] = centerX;
									coords[6][1] = centerY - 1 * step;
									coords[6][2] = 1;
								}
								else
								{
									coords[6][2] = 0;
								}
								if (((centerX + 1 * step >= 0) && (centerX + 1 * step < w)) &&
									((centerY - 1 * step >= 0) && (centerY - 1 * step < h)))
								{
									coords[7][0] = centerX + 1 * step;
									coords[7][1] = centerY - 1 * step;
									coords[7][2] = 1;
								}
								else
								{
									coords[7][2] = 0;
								}
								if (((centerX + 1 * step >= 0) && (centerX + 1 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[8][0] = centerX + 1 * step;
									coords[8][1] = centerY;
									coords[8][2] = 1;
								}
								else
								{
									coords[8][2] = 0;
								}
								if (((centerX + 1 * step >= 0) && (centerX + 1 * step < w)) &&
									((centerY + 1 * step >= 0) && (centerY + 1 * step < h)))
								{
									coords[9][0] = centerX + 1 * step;
									coords[9][1] = centerY + 1 * step;
									coords[9][2] = 1;
								}
								else
								{
									coords[9][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY + 1 * step >= 0) && (centerY + 1 * step < h)))
								{
									coords[10][0] = centerX;
									coords[10][1] = centerY + 1 * step;
									coords[10][2] = 1;
								}
								else
								{
									coords[10][2] = 0;
								}
								if (((centerX - 1 * step >= 0) && (centerX - 1 * step < w)) &&
									((centerY + 1 * step >= 0) && (centerY + 1 * step < h)))
								{
									coords[11][0] = centerX - 1 * step;
									coords[11][1] = centerY + 1 * step;
									coords[11][2] = 1;
								}
								else
								{
									coords[11][2] = 0;
								}
								if (((centerX - 1 * step >= 0) && (centerX - 1 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[12][0] = centerX - 1 * step;
									coords[12][1] = centerY;
									coords[12][2] = 1;
								}
								else
								{
									coords[12][2] = 0;
								}
								MinSAD = 1000000;
								float minCoord[2] = { 10000000,1000000 };
								for (int diamondIndex = 0; diamondIndex < 13; diamondIndex++)
									if (coords[diamondIndex][2] != 0)
									{
										int np = 0;
										float SAD = 0;
										for (int px = -InterpolationSize*BlockSize; px < BlockSize + InterpolationSize*BlockSize; px++)
											for (int py = -InterpolationSize*BlockSize; py < BlockSize + InterpolationSize*BlockSize; py++)
												if (((xBlock*BlockSize + px) >= 0) && ((xBlock*BlockSize + px)<w))
													if (((yBlock*BlockSize + py) >= 0) && ((yBlock*BlockSize + py)<h))
														if (((coords[diamondIndex][0] + px) >= 0) && ((coords[diamondIndex][0] + px)<w))
															if (((coords[diamondIndex][1] + py) >= 0) && ((coords[diamondIndex][1] + py)<h)) {
																RgbaF *thisPt = &pixels.pixelsBeauty[0][0][0] + xBlock*BlockSize + px + (yBlock*BlockSize + py)*w;
																RgbaF *meanPt = &pixels.pixelsBeauty[frame][0][0] + coords[diamondIndex][0] + px + (coords[diamondIndex][1] + py)*w;
																SAD += (thisPt->r - meanPt->r)*(thisPt->r - meanPt->r) + (thisPt->g - meanPt->g)*(thisPt->g - meanPt->g) + (thisPt->b - meanPt->b)*(thisPt->b - meanPt->b);
																np++;
															}
										SAD /= np;
										if (SAD < MinSAD) {
											MinSAD = SAD;
											minCoord[0] = coords[diamondIndex][0];
											minCoord[1] = coords[diamondIndex][1];
										}
									}

								resultCoord[0] = minCoord[0];
								resultCoord[1] = minCoord[1];
								centerX = minCoord[0];
								centerY = minCoord[1];
							}

						if ((alpha > 0) && (MinSAD < Treshold)) {
							blockMV[frame][xBlock][yBlock].r = resultCoord[0] - xBlock*BlockSize;
							blockMV[frame][xBlock][yBlock].g = resultCoord[1] - yBlock*BlockSize;
							blockMV[frame][xBlock][yBlock].b = MinSAD;
							blockMV[frame][xBlock][yBlock].a = 1;
						}
					}
}

void searchBlockDiamond4StepPW(int core)
{
	float Treshold = kernelOpt.tPwSpaceTreshold;
	int searchRadius = kernelOpt.tPwSearchRadius;
	int Iteration = kernelOpt.tPwIterations;

	int w = iOpt.with;
	int h = iOpt.height;
	int div = 1;
	int Pcount = 1;
	float dcount = 1;

	int nframes = 0;
	for (int i = 1; i < iOpt.nFrames; i++)
		if (iOpt.existsFrame[i] == 1)
			nframes++;

	int CoreSize = int(w / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int xBlock = startX; xBlock < endX; xBlock++)
				for (int yBlock = 0; yBlock < h; yBlock++)
					if (xBlock < w) {

						int coords[13][3];
						int centerX = xBlock;
						int centerY = yBlock;
						int resultCoord[2];
						resultCoord[0] = 1000;
						resultCoord[1] = 1000;
						Rgba *bmvPt = &blockMVPosition[frame][0][0] + std::min(std::max(xBlock, 0), w - 1) + std::min(std::max(yBlock, 0), h - 1)*w;

						float MinSAD = 1000000;
						bool isSearch = false;
						float alpha = 0;
						RgbaF *thisPt1 = &pixels.pixelsBeauty[frame][0][0] + std::min(std::max(xBlock, 0), w - 1) + std::min(std::max(yBlock, 0), h - 1)*w;
						alpha = thisPt1->a;

						if (alpha > 0)
							for (int iteration = 0; iteration < Iteration; iteration++) {
								coords[0][0] = centerX;
								coords[0][1] = centerY;
								coords[0][2] = 1;
								int step = (Iteration - iteration) * 1;
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY - 2 * step >= 0) && (centerY - 2 * step < h)))
								{
									coords[1][0] = centerX;
									coords[1][1] = centerY - 2 * step;
									coords[1][2] = 1;
								}
								else
								{
									coords[1][2] = 0;
								}
								if (((centerX + 2 * step >= 0) && (centerX + 2 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[2][0] = centerX + 2 * step;
									coords[2][1] = centerY;
									coords[2][2] = 1;
								}
								else
								{
									coords[2][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY + 2 * step >= 0) && (centerY + 2 * step < h)))
								{
									coords[3][0] = centerX;
									coords[3][1] = centerY + 2 * step;
									coords[3][2] = 1;
								}
								else
								{
									coords[3][2] = 0;
								}
								if (((centerX - 2 * step >= 0) && (centerX - 2 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[4][0] = centerX - 2 * step;
									coords[4][1] = centerY;
									coords[4][2] = 1;
								}
								else
								{
									coords[4][2] = 0;
								}
								if (((centerX - 1 >= 0) && (centerX - 1 * step < w)) &&
									((centerY - 1 >= 0) && (centerY - 1 * step < h)))
								{
									coords[5][0] = centerX - 1 * step;
									coords[5][1] = centerY - 1 * step;
									coords[5][2] = 1;
								}
								else
								{
									coords[5][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY - 1 * step >= 0) && (centerY - 1 * step < h)))
								{
									coords[6][0] = centerX;
									coords[6][1] = centerY - 1 * step;
									coords[6][2] = 1;
								}
								else
								{
									coords[6][2] = 0;
								}
								if (((centerX + 1 * step >= 0) && (centerX + 1 * step < w)) &&
									((centerY - 1 * step >= 0) && (centerY - 1 * step < h)))
								{
									coords[7][0] = centerX + 1 * step;
									coords[7][1] = centerY - 1 * step;
									coords[7][2] = 1;
								}
								else
								{
									coords[7][2] = 0;
								}
								if (((centerX + 1 * step >= 0) && (centerX + 1 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[8][0] = centerX + 1 * step;
									coords[8][1] = centerY;
									coords[8][2] = 1;
								}
								else
								{
									coords[8][2] = 0;
								}
								if (((centerX + 1 * step >= 0) && (centerX + 1 * step < w)) &&
									((centerY + 1 * step >= 0) && (centerY + 1 * step < h)))
								{
									coords[9][0] = centerX + 1 * step;
									coords[9][1] = centerY + 1 * step;
									coords[9][2] = 1;
								}
								else
								{
									coords[9][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY + 1 * step >= 0) && (centerY + 1 * step < h)))
								{
									coords[10][0] = centerX;
									coords[10][1] = centerY + 1 * step;
									coords[10][2] = 1;
								}
								else
								{
									coords[10][2] = 0;
								}
								if (((centerX - 1 * step >= 0) && (centerX - 1 * step < w)) &&
									((centerY + 1 * step >= 0) && (centerY + 1 * step < h)))
								{
									coords[11][0] = centerX - 1 * step;
									coords[11][1] = centerY + 1 * step;
									coords[11][2] = 1;
								}
								else
								{
									coords[11][2] = 0;
								}
								if (((centerX - 1 * step >= 0) && (centerX - 1 * step < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[12][0] = centerX - 1 * step;
									coords[12][1] = centerY;
									coords[12][2] = 1;
								}
								else
								{
									coords[12][2] = 0;
								}

								MinSAD = 1000000;
								float SAD = 0;
								float minCoord[2] = { 10000000,1000000 };
								for (int diamondIndex = 0; diamondIndex < 13; diamondIndex++)
									if (coords[diamondIndex][2] != 0)
									{
										int np = 0;

										RgbaF *thisPt = &pixels.pixelsPosition[0][0][0] + std::min(std::max(xBlock, 0), w - 1) + std::min(std::max(yBlock, 0), h - 1)*w;
										RgbaF *meanPt = &pixels.pixelsPosition[frame][0][0] + std::min(std::max(coords[diamondIndex][0], 0), w - 1) + std::min(std::max(coords[diamondIndex][1], 0), h - 1)*w;

										SAD = (thisPt->r - meanPt->r)*(thisPt->r - meanPt->r) + (thisPt->g - meanPt->g)*(thisPt->g - meanPt->g) + (thisPt->b - meanPt->b)*(thisPt->b - meanPt->b);


										if (SAD < MinSAD) {
											MinSAD = SAD;
											minCoord[0] = coords[diamondIndex][0];
											minCoord[1] = coords[diamondIndex][1];
										}
									}

								resultCoord[0] = minCoord[0];
								resultCoord[1] = minCoord[1];
								centerX = minCoord[0];
								centerY = minCoord[1];
							}
						if (alpha > 0) {
							centerX = resultCoord[0];
							centerY = resultCoord[1];
							float SAD = 0;
							float minCoord[2] = { 10000000,1000000 };
							MinSAD = 1000000;
							for (int x = -searchRadius; x < searchRadius + 1; x++)
								for (int y = -searchRadius; y < searchRadius + 1; y++)
								{
									RgbaF *thisPt = &pixels.pixelsPosition[0][0][0] + std::min(std::max(xBlock, 0), w - 1) + std::min(std::max(yBlock, 0), h - 1)*w;
									RgbaF *meanPt = &pixels.pixelsPosition[frame][0][0] + std::min(std::max(centerX + x, 0), w - 1) + std::min(std::max(centerY + y, 0), h - 1)*w;
									SAD = (thisPt->r - meanPt->r)*(thisPt->r - meanPt->r) + (thisPt->g - meanPt->g)*(thisPt->g - meanPt->g) + (thisPt->b - meanPt->b)*(thisPt->b - meanPt->b);

									if (SAD < MinSAD) {
										MinSAD = SAD;
										resultCoord[0] = centerX + x;
										resultCoord[1] = centerY + y;
									}
								}
						}
						if ((alpha > 0) && (MinSAD < Treshold) && (resultCoord[0] < w) && (resultCoord[1] < h) && (resultCoord[0] >= 0) && (resultCoord[1] >= 0)) {
							bmvPt = &blockMVPosition[frame][0][0] + xBlock + yBlock*w;
							bmvPt->r = resultCoord[0] - xBlock;
							bmvPt->g = resultCoord[1] - yBlock;
							bmvPt->b = MinSAD;
							bmvPt->a = 1;
						}
						else
						{

							bmvPt->r = 0;
							bmvPt->g = 0;
							bmvPt->b = 0;
							bmvPt->a = 0;
						}

						if (alpha == 0) {
							bmvPt->r = 0;
							bmvPt->g = 0;
							bmvPt->b = 0;
							bmvPt->a = 0;
						} 
					}
}

void computeMV(int x, int y, Array<Array2D<Rgba>> &pixelsMotion, imageOptions &iOpt, Array<Rgba> &pPixMV)
{
	int w = iOpt.with;
	int h = iOpt.height;
	int mvr = 0;
	int mvg = 0;

	// forward mv
	for (int i = 0; i < (iOpt.nFrames - 1) / 2 + 1; i++) {
		pPixMV[i].r = 0;
		pPixMV[i].g = 0;
		pPixMV[i].b = 0;
		pPixMV[i].a = 0;
		if (iOpt.existsFrame[i] == 1)
			if (i == 0) {
				Rgba *mvPtr = &pixelsMotion[i][0][0] + x + y*w;
				pPixMV[i].r = mvPtr->r;
				pPixMV[i].g = mvPtr->g;
				pPixMV[i].b = 0;
				pPixMV[i].a = 1;

			}
			else
			{
				mvr = 0;
				mvg = 0;
				float fmvr;
				float fmvg;
				fmvr = float(pPixMV[i - 1].r);
				fmvg = float(pPixMV[i - 1].g);
				if (fmvr > 0) {
					mvr = (int)(fmvr + 0.5);
				}
				else
				{
					mvr = (int)(fmvr - 0.5);
				}
				if (fmvg > 0) {
					mvg = (int)(fmvg + 0.5);
				}
				else
				{
					mvg = (int)(fmvg - 0.5);
				}
				Rgba *mvPtr = &pixelsMotion[i][0][0] + std::min(std::max(x + mvr, 0), w - 1) + std::min(std::max(y + mvg, 0), h - 1)*w;
				pPixMV[i].r = mvPtr->r + pPixMV[i - 1].r;
				pPixMV[i].g = mvPtr->g + pPixMV[i - 1].g;
				pPixMV[i].b = 0;
				pPixMV[i].a = 1;
			}
	}

	// backward mv
	for (int i = (iOpt.nFrames - 1) / 2 + 1; i < iOpt.nFrames; i++) {
		pPixMV[i].r = 0;
		pPixMV[i].g = 0;
		pPixMV[i].b = 0;
		pPixMV[i].a = 0;
		if (iOpt.existsFrame[i] == 1)
			if (i == ((iOpt.nFrames - 1) / 2 + 1)) {
				Rgba *mvPtr = &pixelsMotion[i][0][0] + std::min(std::max(x, 0), w - 1) + std::min(std::max(y, 0), h - 1)*w;
				pPixMV[i].r = -mvPtr->r;
				pPixMV[i].g = -mvPtr->g;
				pPixMV[i].b = 0;
				pPixMV[i].a = 1;
			}
			else
			{
				mvr = 0;
				mvg = 0;
				float fmvr;
				float fmvg;
				fmvr = float(pPixMV[i - 1].r);
				fmvg = float(pPixMV[i - 1].g);
				if (fmvr > 0) {
					mvr = (int)(fmvr + 0.5);
				}
				else
				{
					mvr = (int)(fmvr - 0.5);
				}
				if (fmvg > 0) {
					mvg = (int)(fmvg + 0.5);
				}
				else
				{
					mvg = (int)(fmvg - 0.5);
				}
				Rgba *mvPtr = &pixelsMotion[i][0][0] + std::min(std::max(x + mvr, 0), w - 1) + std::min(std::max(y + mvg, 0), h - 1)*w;
				pPixMV[i].r = -mvPtr->r + pPixMV[i - 1].r;
				pPixMV[i].g = -mvPtr->g + pPixMV[i - 1].g;
				pPixMV[i].b = 0;
				pPixMV[i].a = 1;
			}
	}
}

void searchFullPw(Array<Array2D<Rgba>> &pixels, imageOptions &iOpt, Array<Array2D<Rgba>> &blockMV, float Treshold, int searchRadius)
{

	int w = iOpt.with;
	int h = iOpt.height;
	int div = 1;
	int Pcount = 1;
	float dcount = 1;
	int resultCoord[2];
	int nframes = 0;
	Array<Rgba> pPixMV(iOpt.nFrames);

	for (int i = 1; i < iOpt.nFrames; i++)
		if (iOpt.existsFrame[i] == 1)
			nframes++;
	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int xBlock = 0; xBlock < w; xBlock++)
				for (int yBlock = 0; yBlock < h; yBlock++) {
					Rgba *bmvPt = &blockMV[frame][0][0] + std::min(std::max(xBlock, 0), w - 1) + std::min(std::max(yBlock, 0), h - 1)*w;
					bmvPt->r = 0;
					bmvPt->g = 0;
					bmvPt->b = 0;
					bmvPt->a = 0;
					float MinSAD = 1000000;
					float alpha = 0;
					Rgba *thisPt1 = &pixels[frame][0][0] + std::min(std::max(xBlock, 0), w - 1) + std::min(std::max(yBlock, 0), h - 1)*w;
					alpha = thisPt1->a;
					if (alpha > 0) {
						MinSAD = 1000000;
						float SAD = 0;
						int minCoord[2] = { 10000000,1000000 };
						int mvr = 0;
						int mvg = 0;
						for (int x = -searchRadius; x < searchRadius + 1; x++)
							for (int y = -searchRadius; y < searchRadius + 1; y++)
							{
								Rgba *thisPt = &pixels[0][0][0] + std::min(std::max(xBlock, 0), w - 1) + std::min(std::max(yBlock, 0), h - 1)*w;
								Rgba *meanPt = &pixels[frame][0][0] + std::min(std::max(xBlock + x, 0), w - 1) + std::min(std::max(yBlock + y, 0), h - 1)*w;
								SAD = (thisPt->r - meanPt->r)*(thisPt->r - meanPt->r) + (thisPt->g - meanPt->g)*(thisPt->g - meanPt->g) + (thisPt->b - meanPt->b)*(thisPt->b - meanPt->b);

								if (SAD < MinSAD) {
									MinSAD = SAD;
									resultCoord[0] = xBlock + x;
									resultCoord[1] = yBlock + y;
								}
							}
					}
					if ((alpha > 0) && (MinSAD < Treshold) && (resultCoord[0] < w) && (resultCoord[1] < h) && (resultCoord[0] >= 0) && (resultCoord[1] >= 0)) {
						bmvPt = &blockMV[frame][0][0] + xBlock + yBlock*w;
						bmvPt->r = resultCoord[0] - xBlock;
						bmvPt->g = resultCoord[1] - yBlock;
						bmvPt->b = MinSAD;
						bmvPt->a = 1;
					}

					if (dcount != ((Pcount * 100) / (nframes*(w)*(h))))
						printf("\rCompute motion compensation %d%% complete.", (Pcount * 100) / (nframes*(w)*(h)));

					dcount = (Pcount * 100) / (nframes*(w)*(h));
					Pcount++;
				}
}

void BlockSmoothPw(Array<Array2D<Rgba>> &blockMV, Array<Array2D<RgbaF>> &ResultBlockSmooth, imageOptions &iOpt, imgPixels &pixels)
{
	int w = iOpt.with;
	int h = iOpt.height;

	RgbaF *pixBeauty = &pixels.pixelsBeauty[0][0][0];
	RgbaF *ResultBPix = &ResultBlockSmooth[0][0][0];
	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = 0; bx < w; bx++)
				for (int by = 0; by < h; by++) {
					int mvg, mvr;
					Rgba *bmvPt = &blockMV[frame][0][0] + bx + by*w;
					mvr = int(bmvPt->r);
					mvg = int(bmvPt->g);

					pixBeauty = &pixels.pixelsBeauty[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultBPix = &ResultBlockSmooth[frame][0][0] + bx + by*iOpt.with;

					ResultBPix->r = pixBeauty->r;
					ResultBPix->g = pixBeauty->g;
					ResultBPix->b = pixBeauty->b;
					ResultBPix->a = bmvPt->a;

				}
}

void BlockSmoothPW2(int core)
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;


	RgbaF *pixBeauty;
	RgbaF *ResultBPix;

	RgbaF *pixAlbedo;
	RgbaF *ResultAlbedoPix;

	RgbaF *pixSpec;
	RgbaF *ResultSpecPix;

	RgbaF *pixDiffuse;
	RgbaF *ResultDiffusePix;

	RgbaF *pixIndDiffuse;
	RgbaF *ResultIndDiffusePix;

	RgbaF *pixIndSpec;
	RgbaF *ResultIndSpecPix;

	RgbaF *pixRefract;
	RgbaF *ResultRefractPix;

	//tech channel
	RgbaF *pixDepth;
	RgbaF *ResultDepthPix;

	RgbaF *pixNormal;
	RgbaF *ResultNormalPix;

	RgbaF *pixPosition;
	RgbaF *ResultPositionPix;

	Rgba *tSWeightPix;
	Rgba *tDWeightPix;

	int CoreSize = int(wResize / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int frame = 0; frame < 1; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = startX; bx < endX; bx++)
				for (int by = 0; by < hResize; by++)
					if (bx < wResize) {
						tSWeightPix = &tSWeight[0][0] + bx + by*iOpt.with;
						tDWeightPix = &tDWeight[0][0] + bx + by*iOpt.with;

						pixBeauty = &pixels.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						ResultBPix = &ResultBlockSmoothTemporal.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;

						pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
						ResultAlbedoPix = &ResultBlockSmoothTemporal.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;

						pixSpec = &pixels.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;
						ResultSpecPix = &ResultBlockSmoothTemporal.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

						pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						ResultDiffusePix = &ResultBlockSmoothTemporal.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;


						pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						ResultIndDiffusePix = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;

						pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						ResultIndSpecPix = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;


						pixRefract = &pixels.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						ResultRefractPix = &ResultBlockSmoothTemporal.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;

						//tech channel
						pixDepth = &pixels.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
						ResultDepthPix = &ResultBlockSmoothTemporal.pixelsDepth[frame][0][0] + bx + by*iOpt.with;

						pixNormal = &pixels.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
						ResultNormalPix = &ResultBlockSmoothTemporal.pixelsNormal[frame][0][0] + bx + by*iOpt.with;

						pixPosition = &pixels.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
						ResultPositionPix = &ResultBlockSmoothTemporal.pixelsPosition[frame][0][0] + bx + by*iOpt.with;

						ResultBPix->a = pixBeauty->a;
						ResultBPix->r = pixBeauty->r;
						ResultBPix->g = pixBeauty->g;
						ResultBPix->b = pixBeauty->b;

						ResultAlbedoPix->r = pixAlbedo->r;
						ResultAlbedoPix->g = pixAlbedo->g;
						ResultAlbedoPix->b = pixAlbedo->b;

						ResultSpecPix->r = pixSpec->r;
						ResultSpecPix->g = pixSpec->g;
						ResultSpecPix->b = pixSpec->b;

						ResultDiffusePix->r = pixDiffuse->r;
						ResultDiffusePix->g = pixDiffuse->g;
						ResultDiffusePix->b = pixDiffuse->b;

						ResultIndDiffusePix->r = pixIndDiffuse->r;
						ResultIndDiffusePix->g = pixIndDiffuse->g;
						ResultIndDiffusePix->b = pixIndDiffuse->b;

						ResultIndSpecPix->r = pixIndSpec->r;
						ResultIndSpecPix->g = pixIndSpec->g;
						ResultIndSpecPix->b = pixIndSpec->b;

						ResultRefractPix->r = pixRefract->r;
						ResultRefractPix->g = pixRefract->g;
						ResultRefractPix->b = pixRefract->b;

						//tech channel
						ResultDepthPix->r = pixDepth->r;
						ResultDepthPix->g = pixDepth->g;
						ResultDepthPix->b = pixDepth->b;

						ResultNormalPix->r = pixNormal->r;
						ResultNormalPix->g = pixNormal->g;
						ResultNormalPix->b = pixNormal->b;

						ResultPositionPix->r = pixPosition->r;
						ResultPositionPix->g = pixPosition->g;
						ResultPositionPix->b = pixPosition->b;
					}
	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = startX; bx < endX; bx++)
				for (int by = 0; by < hResize; by++)
					if (bx < wResize) {
						int mvg, mvr;
						Rgba *bmvPt = &blockMVPosition[frame][0][0] + bx + by*iOpt.with;
						mvr = int(bmvPt->r);
						mvg = int(bmvPt->g);

						tSWeightPix = &tSWeight[0][0] + bx + by*iOpt.with;
						tDWeightPix = &tDWeight[0][0] + bx + by*iOpt.with;

						pixBeauty = &pixels.pixelsBeauty[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultBPix = &ResultBlockSmoothTemporal.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;


						pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultAlbedoPix = &ResultBlockSmoothTemporal.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;

						pixSpec = &pixels.pixelsSpecular[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultSpecPix = &ResultBlockSmoothTemporal.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

						pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultDiffusePix = &ResultBlockSmoothTemporal.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;

						pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultIndDiffusePix = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;

						pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultIndSpecPix = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;

						pixRefract = &pixels.pixelsRefraction[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultRefractPix = &ResultBlockSmoothTemporal.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;

						//tech
						pixDepth = &pixels.pixelsDepth[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultDepthPix = &ResultBlockSmoothTemporal.pixelsDepth[frame][0][0] + bx + by*iOpt.with;

						pixNormal = &pixels.pixelsNormal[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultNormalPix = &ResultBlockSmoothTemporal.pixelsNormal[frame][0][0] + bx + by*iOpt.with;

						pixPosition = &pixels.pixelsPosition[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
						ResultPositionPix = &ResultBlockSmoothTemporal.pixelsPosition[frame][0][0] + bx + by*iOpt.with;

						float weight;
						weight = bmvPt->a;

						ResultBPix->a = weight;
						ResultBPix->r = pixBeauty->r;
						ResultBPix->g = pixBeauty->g;
						ResultBPix->b = pixBeauty->b;

						ResultAlbedoPix->r = pixAlbedo->r;
						ResultAlbedoPix->g = pixAlbedo->g;
						ResultAlbedoPix->b = pixAlbedo->b;

						ResultSpecPix->r = pixSpec->r;
						ResultSpecPix->g = pixSpec->g;
						ResultSpecPix->b = pixSpec->b;

						ResultDiffusePix->r = pixDiffuse->r;
						ResultDiffusePix->g = pixDiffuse->g;
						ResultDiffusePix->b = pixDiffuse->b;

						ResultIndDiffusePix->r = pixIndDiffuse->r;
						ResultIndDiffusePix->g = pixIndDiffuse->g;
						ResultIndDiffusePix->b = pixIndDiffuse->b;

						ResultIndSpecPix->r = pixIndSpec->r;
						ResultIndSpecPix->g = pixIndSpec->g;
						ResultIndSpecPix->b = pixIndSpec->b;

						ResultRefractPix->r = pixRefract->r;
						ResultRefractPix->g = pixRefract->g;
						ResultRefractPix->b = pixRefract->b;

						//tech channel
						ResultDepthPix->r = pixDepth->r;
						ResultDepthPix->g = pixDepth->g;
						ResultDepthPix->b = pixDepth->b;

						ResultNormalPix->r = pixNormal->r;
						ResultNormalPix->g = pixNormal->g;
						ResultNormalPix->b = pixNormal->b;

						ResultPositionPix->r = pixPosition->r;
						ResultPositionPix->g = pixPosition->g;
						ResultPositionPix->b = pixPosition->b;
					}
}

void BlockSmoothPW2_coreNo()
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;

	RgbaF *pixBeauty;
	RgbaF *pixBeautyOrig;
	RgbaF *ResultBPix;

	RgbaF *pixPos;
	RgbaF *pixPosOrig;
	RgbaF *ResultPosPix;

	RgbaF *pixAlbedo;
	RgbaF *pixAlbedoOrig;
	RgbaF *ResultAlbedoPix;

	RgbaF *pixSpec;
	RgbaF *ResultSpecPix;

	RgbaF *pixDiffuse;
	RgbaF *ResultDiffusePix;

	RgbaF *pixIndDiffuse;
	RgbaF *ResultIndDiffusePix;

	RgbaF *pixIndSpec;
	RgbaF *ResultIndSpecPix;

	RgbaF *pixRefract;
	RgbaF *ResultRefractPix;

	//tech channel
	RgbaF *pixDepth;
	RgbaF *ResultDepthPix;

	RgbaF *pixNormal;
	RgbaF *ResultNormalPix;

	RgbaF *pixPosition;
	RgbaF *ResultPositionPix;

	Rgba *tSWeightPix;
	Rgba *tDWeightPix;

	for (int frame = 0; frame < 1; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = 0; bx < wResize; bx++)
				for (int by = 0; by < hResize; by++) {

					tSWeightPix = &tSWeight[0][0] + bx + by*iOpt.with;
					tDWeightPix = &tDWeight[0][0] + bx + by*iOpt.with;

					pixBeauty = &pixels.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
					ResultBPix = &ResultBlockSmoothTemporal.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;


					pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
					ResultAlbedoPix = &ResultBlockSmoothTemporal.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;

					pixSpec = &pixels.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;
					ResultSpecPix = &ResultBlockSmoothTemporal.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

					pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
					ResultDiffusePix = &ResultBlockSmoothTemporal.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;

					pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
					ResultIndDiffusePix = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;

					pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
					ResultIndSpecPix = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;

					pixRefract = &pixels.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
					ResultRefractPix = &ResultBlockSmoothTemporal.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;

					//tech channel
					pixDepth = &pixels.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
					ResultDepthPix = &ResultBlockSmoothTemporal.pixelsDepth[frame][0][0] + bx + by*iOpt.with;

					pixNormal = &pixels.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
					ResultNormalPix = &ResultBlockSmoothTemporal.pixelsNormal[frame][0][0] + bx + by*iOpt.with;

					pixPosition = &pixels.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
					ResultPositionPix = &ResultBlockSmoothTemporal.pixelsPosition[frame][0][0] + bx + by*iOpt.with;

					ResultBPix->a = pixBeauty->a;
					ResultBPix->r = pixBeauty->r;
					ResultBPix->g = pixBeauty->g;
					ResultBPix->b = pixBeauty->b;

					ResultAlbedoPix->r = pixAlbedo->r;
					ResultAlbedoPix->g = pixAlbedo->g;
					ResultAlbedoPix->b = pixAlbedo->b;

					ResultSpecPix->r = pixSpec->r;
					ResultSpecPix->g = pixSpec->g;
					ResultSpecPix->b = pixSpec->b;

					ResultDiffusePix->r = pixDiffuse->r;
					ResultDiffusePix->g = pixDiffuse->g;
					ResultDiffusePix->b = pixDiffuse->b;

					ResultIndDiffusePix->r = pixIndDiffuse->r;
					ResultIndDiffusePix->g = pixIndDiffuse->g;
					ResultIndDiffusePix->b = pixIndDiffuse->b;

					ResultIndSpecPix->r = pixIndSpec->r;
					ResultIndSpecPix->g = pixIndSpec->g;
					ResultIndSpecPix->b = pixIndSpec->b;

					ResultRefractPix->r = pixRefract->r;
					ResultRefractPix->g = pixRefract->g;
					ResultRefractPix->b = pixRefract->b;

					//tech channel
					ResultDepthPix->r = pixDepth->r;
					ResultDepthPix->g = pixDepth->g;
					ResultDepthPix->b = pixDepth->b;

					ResultNormalPix->r = pixNormal->r;
					ResultNormalPix->g = pixNormal->g;
					ResultNormalPix->b = pixNormal->b;

					ResultPositionPix->r = pixPosition->r;
					ResultPositionPix->g = pixPosition->g;
					ResultPositionPix->b = pixPosition->b;
				}
	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = 0; bx < wResize; bx++)
				for (int by = 0; by < hResize; by++) {
					int mvg, mvr;
					Rgba *bmvPt = &blockMVPosition[frame][0][0] + bx + by*iOpt.with;
					mvr = int(bmvPt->r);
					mvg = int(bmvPt->g);

					tSWeightPix = &tSWeight[0][0] + bx + by*iOpt.with;
					tDWeightPix = &tDWeight[0][0] + bx + by*iOpt.with;

					pixBeauty = &pixels.pixelsBeauty[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultBPix = &ResultBlockSmoothTemporal.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;

					pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultAlbedoPix = &ResultBlockSmoothTemporal.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;

					pixSpec = &pixels.pixelsSpecular[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultSpecPix = &ResultBlockSmoothTemporal.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

					pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultDiffusePix = &ResultBlockSmoothTemporal.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;

					pixIndDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultIndDiffusePix = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;

					pixIndSpec = &pixels.pixelsIndirectSpecular[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultIndSpecPix = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;

					pixRefract = &pixels.pixelsRefraction[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultRefractPix = &ResultBlockSmoothTemporal.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;

					//tech
					pixDepth = &pixels.pixelsDepth[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultDepthPix = &ResultBlockSmoothTemporal.pixelsDepth[frame][0][0] + bx + by*iOpt.with;

					pixNormal = &pixels.pixelsNormal[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultNormalPix = &ResultBlockSmoothTemporal.pixelsNormal[frame][0][0] + bx + by*iOpt.with;

					pixPosition = &pixels.pixelsPosition[frame][0][0] + std::min(std::max(bx + mvr, 0), iOpt.with - 1) + std::min(std::max(by + mvg, 0), iOpt.height - 1)*iOpt.with;
					ResultPositionPix = &ResultBlockSmoothTemporal.pixelsPosition[frame][0][0] + bx + by*iOpt.with;

					float weight;
					weight = bmvPt->a;
					ResultBPix->a = weight;
					ResultBPix->r = ResultBPix->r*(1 - weight) + pixBeauty->r*weight;
					ResultBPix->g = ResultBPix->g*(1 - weight) + pixBeauty->g*weight;
					ResultBPix->b = ResultBPix->b*(1 - weight) + pixBeauty->b*weight;

					ResultAlbedoPix->r = ResultAlbedoPix->r*(1 - weight) + pixAlbedo->r*weight;
					ResultAlbedoPix->g = ResultAlbedoPix->g*(1 - weight) + pixAlbedo->g*weight;
					ResultAlbedoPix->b = ResultAlbedoPix->b*(1 - weight) + pixAlbedo->b*weight;

					ResultSpecPix->r = ResultSpecPix->r*(1 - weight) + pixSpec->r*weight;
					ResultSpecPix->g = ResultSpecPix->g*(1 - weight) + pixSpec->g*weight;
					ResultSpecPix->b = ResultSpecPix->b*(1 - weight) + pixSpec->b*weight;

					ResultDiffusePix->r = ResultDiffusePix->r*(1 - weight) + pixDiffuse->r*weight;
					ResultDiffusePix->g = ResultDiffusePix->g*(1 - weight) + pixDiffuse->g*weight;
					ResultDiffusePix->b = ResultDiffusePix->b*(1 - weight) + pixDiffuse->b*weight;

					ResultIndDiffusePix->r = ResultIndDiffusePix->r*(1 - weight) + pixIndDiffuse->r*weight;
					ResultIndDiffusePix->g = ResultIndDiffusePix->g*(1 - weight) + pixIndDiffuse->g*weight;
					ResultIndDiffusePix->b = ResultIndDiffusePix->b*(1 - weight) + pixIndDiffuse->b*weight;

					ResultIndSpecPix->r = ResultIndSpecPix->r*(1 - weight) + pixIndSpec->r*weight;
					ResultIndSpecPix->g = ResultIndSpecPix->g*(1 - weight) + pixIndSpec->g*weight;
					ResultIndSpecPix->b = ResultIndSpecPix->b*(1 - weight) + pixIndSpec->b*weight;

					ResultRefractPix->r = ResultRefractPix->r*(1 - weight) + pixRefract->r*weight;
					ResultRefractPix->g = ResultRefractPix->g*(1 - weight) + pixRefract->g*weight;
					ResultRefractPix->b = ResultRefractPix->b*(1 - weight) + pixRefract->b*weight;

					//tech channel
					ResultDepthPix->r = ResultDepthPix->r*(1 - weight) + pixDepth->r*weight;
					ResultDepthPix->g = ResultDepthPix->g*(1 - weight) + pixDepth->g*weight;
					ResultDepthPix->b = ResultDepthPix->b*(1 - weight) + pixDepth->b*weight;

					ResultNormalPix->r = ResultNormalPix->r*(1 - weight) + pixNormal->r*weight;
					ResultNormalPix->g = ResultNormalPix->g*(1 - weight) + pixNormal->g*weight;
					ResultNormalPix->b = ResultNormalPix->b*(1 - weight) + pixNormal->b*weight;

					ResultPositionPix->r = ResultPositionPix->r*(1 - weight) + pixPosition->r*weight;
					ResultPositionPix->g = ResultPositionPix->g*(1 - weight) + pixPosition->g*weight;
					ResultPositionPix->b = ResultPositionPix->b*(1 - weight) + pixPosition->b*weight;
				}
}

void searchBlockFull(Array<Array2D<Rgba>> &pixels, imageOptions &iOpt, Array<Array2D<Rgba>> &blockMV, int BlockSize, float InterpolationSize, float Treshold)
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;
	int w = iOpt.with;
	int h = iOpt.height;
	int div = 1;
	int Pcount = 1;
	float dcount = 1;

	while (div != 0) {
		div = wResize % BlockSize;
		if (div != 0)
			wResize++;
	}
	div = 1;
	while (div != 0) {
		div = hResize % BlockSize;
		if (div != 0)
			hResize++;
	}

	int nframes = 0;
	for (int i = 1; i < iOpt.nFrames; i++)
		if (iOpt.existsFrame[i] == 1)
			nframes++;
	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int xBlock = 0; xBlock < wResize / BlockSize; xBlock++)
				for (int yBlock = 0; yBlock < hResize / BlockSize; yBlock++) {

					int coords[13][3];
					int centerX = xBlock*BlockSize;
					int centerY = yBlock*BlockSize;
					int resultCoord[2];
					resultCoord[0] = 1000;
					resultCoord[1] = 1000;
					blockMV[frame][xBlock][yBlock].r = 0;
					blockMV[frame][xBlock][yBlock].g = 0;
					blockMV[frame][xBlock][yBlock].b = 0;
					blockMV[frame][xBlock][yBlock].a = 0;
					float MinSAD = 1000000;
					bool isSearch = false;
					float alpha = 0;
					for (int xb = xBlock*BlockSize; xb < std::min(xBlock*BlockSize + BlockSize, w - 1); xb++)
						for (int yb = yBlock*BlockSize; yb < std::min(yBlock*BlockSize + BlockSize, h - 1); yb++) {
							Rgba *thisPt = &pixels[frame][0][0] + xb + yb*w;
							alpha += thisPt->a;
						}
					if (alpha > 0)
						while (!isSearch) {
							coords[0][0] = centerX;
							coords[0][1] = centerY;
							coords[0][2] = 1;
							if (((centerX >= 0) && (centerX < w)) &&
								((centerY - 2 >= 0) && (centerY - 2 < h)))
							{
								coords[1][0] = centerX;
								coords[1][1] = centerY - 2;
								coords[1][2] = 1;
							}
							else
							{
								coords[1][2] = 0;
							}
							if (((centerX + 2 >= 0) && (centerX + 2 < w)) &&
								((centerY >= 0) && (centerY < h)))
							{
								coords[2][0] = centerX + 2;
								coords[2][1] = centerY;
								coords[2][2] = 1;
							}
							else
							{
								coords[2][2] = 0;
							}
							if (((centerX >= 0) && (centerX < w)) &&
								((centerY + 2 >= 0) && (centerY + 2 < h)))
							{
								coords[3][0] = centerX;
								coords[3][1] = centerY + 2;
								coords[3][2] = 1;
							}
							else
							{
								coords[3][2] = 0;
							}
							if (((centerX - 2 >= 0) && (centerX - 2 < w)) &&
								((centerY >= 0) && (centerY < h)))
							{
								coords[4][0] = centerX - 2;
								coords[4][1] = centerY;
								coords[4][2] = 1;
							}
							else
							{
								coords[4][2] = 0;
							}
							if (((centerX - 1 >= 0) && (centerX - 1 < w)) &&
								((centerY - 1 >= 0) && (centerY - 1 < h)))
							{
								coords[5][0] = centerX - 1;
								coords[5][1] = centerY - 1;
								coords[5][2] = 1;
							}
							else
							{
								coords[5][2] = 0;
							}
							if (((centerX >= 0) && (centerX < w)) &&
								((centerY - 1 >= 0) && (centerY - 1 < h)))
							{
								coords[6][0] = centerX;
								coords[6][1] = centerY - 1;
								coords[6][2] = 1;
							}
							else
							{
								coords[6][2] = 0;
							}
							if (((centerX + 1 >= 0) && (centerX + 1 < w)) &&
								((centerY - 1 >= 0) && (centerY - 1 < h)))
							{
								coords[7][0] = centerX + 1;
								coords[7][1] = centerY - 1;
								coords[7][2] = 1;
							}
							else
							{
								coords[7][2] = 0;
							}
							if (((centerX + 1 >= 0) && (centerX + 1 < w)) &&
								((centerY >= 0) && (centerY < h)))
							{
								coords[8][0] = centerX + 1;
								coords[8][1] = centerY;
								coords[8][2] = 1;
							}
							else
							{
								coords[8][2] = 0;
							}
							if (((centerX + 1 >= 0) && (centerX + 1 < w)) &&
								((centerY + 1 >= 0) && (centerY + 1 < h)))
							{
								coords[9][0] = centerX + 1;
								coords[9][1] = centerY + 1;
								coords[9][2] = 1;
							}
							else
							{
								coords[9][2] = 0;
							}
							if (((centerX >= 0) && (centerX < w)) &&
								((centerY + 1 >= 0) && (centerY + 1 < h)))
							{
								coords[10][0] = centerX;
								coords[10][1] = centerY + 1;
								coords[10][2] = 1;
							}
							else
							{
								coords[10][2] = 0;
							}
							if (((centerX - 1 >= 0) && (centerX - 1 < w)) &&
								((centerY + 1 >= 0) && (centerY + 1 < h)))
							{
								coords[11][0] = centerX - 1;
								coords[11][1] = centerY + 1;
								coords[11][2] = 1;
							}
							else
							{
								coords[11][2] = 0;
							}
							if (((centerX - 1 >= 0) && (centerX - 1 < w)) &&
								((centerY >= 0) && (centerY < h)))
							{
								coords[12][0] = centerX - 1;
								coords[12][1] = centerY;
								coords[12][2] = 1;
							}
							else
							{
								coords[12][2] = 0;
							}

							MinSAD = 1000000;
							float minCoord[2] = { 10000000,1000000 };
							for (int diamondIndex = 0; diamondIndex < 13; diamondIndex++)
								if (coords[diamondIndex][2] != 0)
								{
									int np = 0;
									float SAD = 0;
									for (int px = -InterpolationSize*BlockSize; px < BlockSize + InterpolationSize*BlockSize; px++)
										for (int py = -InterpolationSize*BlockSize; py < BlockSize + InterpolationSize*BlockSize; py++) {
											Rgba *thisPt = &pixels[0][0][0] + std::min(std::max(xBlock*BlockSize + px, 0), w - 1) + std::min(std::max(yBlock*BlockSize + py, 0), h - 1)*w;
											Rgba *meanPt = &pixels[frame][0][0] + std::min(std::max(coords[diamondIndex][0] + px, 0), w - 1) + std::min(std::max(coords[diamondIndex][1] + py, 0), h - 1)*w;
											SAD += (thisPt->r - meanPt->r)*(thisPt->r - meanPt->r) + (thisPt->g - meanPt->g)*(thisPt->g - meanPt->g) + (thisPt->b - meanPt->b)*(thisPt->b - meanPt->b);
											np++;
										}
									SAD /= np;
									if (SAD < MinSAD) {
										MinSAD = SAD;
										minCoord[0] = coords[diamondIndex][0];
										minCoord[1] = coords[diamondIndex][1];
									}
								}
							if ((minCoord[0] == coords[0][0]) && (minCoord[1] == coords[0][1])) {
								resultCoord[0] = minCoord[0];
								resultCoord[1] = minCoord[1];
								isSearch = true;
							}
							else
							{
								centerX = minCoord[0];
								centerY = minCoord[1];
							}
						}

					if ((alpha > 0) && (MinSAD < Treshold)) {
						blockMV[frame][xBlock][yBlock].r = resultCoord[0] - xBlock*BlockSize;
						blockMV[frame][xBlock][yBlock].g = resultCoord[1] - yBlock*BlockSize;
						blockMV[frame][xBlock][yBlock].b = MinSAD;
						blockMV[frame][xBlock][yBlock].a = 1;
					}
				}

}

void searchBlockDiamond(int core)
{
	int wResize, hResize;
	wResize = iOpt.with;
	hResize = iOpt.height;

	int BlockSize = kernelOpt.tBlockSize;
	float InterpolationSize = kernelOpt.tInterpolation;
	float Treshold = kernelOpt.tColorTreshold;

	int w = iOpt.with;
	int h = iOpt.height;
	int div = 1;
	int Pcount = 1;
	float dcount = 1;

	while (div != 0) {
		div = wResize % BlockSize;
		if (div != 0)
			wResize++;
	}
	div = 1;
	while (div != 0) {
		div = hResize % BlockSize;
		if (div != 0)
			hResize++;
	}

	int nframes = 0;
	for (int i = 1; i < iOpt.nFrames; i++)
		if (iOpt.existsFrame[i] == 1)
			nframes++;

	int CoreSize = int((wResize / BlockSize) / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int frame = 1; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int xBlock = startX; xBlock < endX; xBlock++)
				for (int yBlock = 0; yBlock < hResize / BlockSize; yBlock++)
					if (xBlock < wResize / BlockSize)
					{
						int coords[13][3];
						int centerX = xBlock*BlockSize;
						int centerY = yBlock*BlockSize;
						int resultCoord[2];
						resultCoord[0] = 1000;
						resultCoord[1] = 1000;
						blockMV[frame][xBlock][yBlock].r = 0;
						blockMV[frame][xBlock][yBlock].g = 0;
						blockMV[frame][xBlock][yBlock].b = 0;
						blockMV[frame][xBlock][yBlock].a = 0;
						float MinSAD = 1000000;
						bool isSearch = false;
						float alpha = 0;
						for (int xb = xBlock*BlockSize; xb < std::min(xBlock*BlockSize + BlockSize, w - 1); xb++)
							for (int yb = yBlock*BlockSize; yb < std::min(yBlock*BlockSize + BlockSize, h - 1); yb++) {
								RgbaF *thisPt = &pixels.pixelsBeauty[frame][0][0] + xb + yb*w;
								alpha += thisPt->a;
							}
						if (alpha > 0)
							while (!isSearch) {
								coords[0][0] = centerX;
								coords[0][1] = centerY;
								coords[0][2] = 1;
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY - 2 >= 0) && (centerY - 2 < h)))
								{
									coords[1][0] = centerX;
									coords[1][1] = centerY - 2;
									coords[1][2] = 1;
								}
								else
								{
									coords[1][2] = 0;
								}
								if (((centerX + 2 >= 0) && (centerX + 2 < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[2][0] = centerX + 2;
									coords[2][1] = centerY;
									coords[2][2] = 1;
								}
								else
								{
									coords[2][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY + 2 >= 0) && (centerY + 2 < h)))
								{
									coords[3][0] = centerX;
									coords[3][1] = centerY + 2;
									coords[3][2] = 1;
								}
								else
								{
									coords[3][2] = 0;
								}
								if (((centerX - 2 >= 0) && (centerX - 2 < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[4][0] = centerX - 2;
									coords[4][1] = centerY;
									coords[4][2] = 1;
								}
								else
								{
									coords[4][2] = 0;
								}
								if (((centerX - 1 >= 0) && (centerX - 1 < w)) &&
									((centerY - 1 >= 0) && (centerY - 1 < h)))
								{
									coords[5][0] = centerX - 1;
									coords[5][1] = centerY - 1;
									coords[5][2] = 1;
								}
								else
								{
									coords[5][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY - 1 >= 0) && (centerY - 1 < h)))
								{
									coords[6][0] = centerX;
									coords[6][1] = centerY - 1;
									coords[6][2] = 1;
								}
								else
								{
									coords[6][2] = 0;
								}
								if (((centerX + 1 >= 0) && (centerX + 1 < w)) &&
									((centerY - 1 >= 0) && (centerY - 1 < h)))
								{
									coords[7][0] = centerX + 1;
									coords[7][1] = centerY - 1;
									coords[7][2] = 1;
								}
								else
								{
									coords[7][2] = 0;
								}
								if (((centerX + 1 >= 0) && (centerX + 1 < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[8][0] = centerX + 1;
									coords[8][1] = centerY;
									coords[8][2] = 1;
								}
								else
								{
									coords[8][2] = 0;
								}
								if (((centerX + 1 >= 0) && (centerX + 1 < w)) &&
									((centerY + 1 >= 0) && (centerY + 1 < h)))
								{
									coords[9][0] = centerX + 1;
									coords[9][1] = centerY + 1;
									coords[9][2] = 1;
								}
								else
								{
									coords[9][2] = 0;
								}
								if (((centerX >= 0) && (centerX < w)) &&
									((centerY + 1 >= 0) && (centerY + 1 < h)))
								{
									coords[10][0] = centerX;
									coords[10][1] = centerY + 1;
									coords[10][2] = 1;
								}
								else
								{
									coords[10][2] = 0;
								}
								if (((centerX - 1 >= 0) && (centerX - 1 < w)) &&
									((centerY + 1 >= 0) && (centerY + 1 < h)))
								{
									coords[11][0] = centerX - 1;
									coords[11][1] = centerY + 1;
									coords[11][2] = 1;
								}
								else
								{
									coords[11][2] = 0;
								}
								if (((centerX - 1 >= 0) && (centerX - 1 < w)) &&
									((centerY >= 0) && (centerY < h)))
								{
									coords[12][0] = centerX - 1;
									coords[12][1] = centerY;
									coords[12][2] = 1;
								}
								else
								{
									coords[12][2] = 0;
								}

								MinSAD = 1000000;
								float minCoord[2] = { 10000000,1000000 };
								for (int diamondIndex = 0; diamondIndex < 13; diamondIndex++)
									if (coords[diamondIndex][2] != 0)
									{
										int np = 0;
										float SAD = 0;
										for (int px = -InterpolationSize*BlockSize; px < BlockSize + InterpolationSize*BlockSize; px++)
											for (int py = -InterpolationSize*BlockSize; py < BlockSize + InterpolationSize*BlockSize; py++)
												if (((xBlock*BlockSize + px) >= 0) && ((xBlock*BlockSize + px)<w))
													if (((yBlock*BlockSize + py) >= 0) && ((yBlock*BlockSize + py)<h))
														if (((coords[diamondIndex][0] + px) >= 0) && ((coords[diamondIndex][0] + px)<w))
															if (((coords[diamondIndex][1] + py) >= 0) && ((coords[diamondIndex][1] + py)<h)) {
																RgbaF *thisPt = &pixels.pixelsBeauty[0][0][0] + xBlock*BlockSize + px + (yBlock*BlockSize + py)*w;
																RgbaF *meanPt = &pixels.pixelsBeauty[frame][0][0] + coords[diamondIndex][0] + px + (coords[diamondIndex][1] + py)*w;
																SAD += (thisPt->r - meanPt->r)*(thisPt->r - meanPt->r) + (thisPt->g - meanPt->g)*(thisPt->g - meanPt->g) + (thisPt->b - meanPt->b)*(thisPt->b - meanPt->b);
																np++;
															}
										SAD /= np;
										if (SAD < MinSAD) {
											MinSAD = SAD;
											minCoord[0] = coords[diamondIndex][0];
											minCoord[1] = coords[diamondIndex][1];
										}
									}
								if ((minCoord[0] == coords[0][0]) && (minCoord[1] == coords[0][1])) {
									resultCoord[0] = minCoord[0];
									resultCoord[1] = minCoord[1];
									isSearch = true;
								}
								else
								{
									centerX = minCoord[0];
									centerY = minCoord[1];
								}
							}

						if ((alpha > 0) && (MinSAD < Treshold)) {
							blockMV[frame][xBlock][yBlock].r = resultCoord[0] - xBlock*BlockSize;
							blockMV[frame][xBlock][yBlock].g = resultCoord[1] - yBlock*BlockSize;
							blockMV[frame][xBlock][yBlock].b = MinSAD;
							blockMV[frame][xBlock][yBlock].a = 1;
						}
						if (alpha == 0) {
							blockMV[frame][xBlock][yBlock].r = 0;
							blockMV[frame][xBlock][yBlock].g = 0;
							blockMV[frame][xBlock][yBlock].b = 0;
							blockMV[frame][xBlock][yBlock].a = 0;
						} 
					}
}

typedef struct _thread_data_t {
	int frame;
	int core;
} thread_data_t;

void computeMSE(Array2D<RgbaF> input, Array2D<RgbaF> out, imageOptions &iOpt)
{
	for (int x = 0; x < iOpt.with; ++x)
		for (int y = 0; y < iOpt.height; ++y) {

		}
}

void NLM_Temporal_Filter(imgPixels &sigmaPass, imgPixels &ResultPass, imgPixels &OutPass, int kernel, int radius, float sigmaColor, float sigmaAlbedo, float sigmaNormal, float sigmaDepth, float sigmaPosition, float SpecStrength, int nFrames, int AD, float epsilon, float firefly, float fallof, imageOptions &iOpt, int core)
{
	int w = iOpt.with;
	int h = iOpt.height;
	int Pcount = 1;
	float dcount = 1;
	Array2D<RgbaF> pix_resSpecularS(iOpt.with, iOpt.height);
	Array2D<RgbaF> pix_resAlbedoS(iOpt.with, iOpt.height);
	Array2D<RgbaF> pix_resDiffuseS(iOpt.with, iOpt.height);
	Array2D<RgbaF> pix_resIndDiffuseS(iOpt.with, iOpt.height);
	Array2D<RgbaF> pix_resIndSpecularS(iOpt.with, iOpt.height);
	Array2D<RgbaF> pix_resRefractionS(iOpt.with, iOpt.height);
	Array2D<RgbaF> pix_resSpecPs(iOpt.with, iOpt.height);
	Array2D<RgbaF> pix_resDiffusePs(iOpt.with, iOpt.height);

	int CoreSize = int(w / iOpt.nCores) + 1;
	int startX = std::max(std::min(CoreSize*core, w), 0);
	int endX = std::max(std::min(CoreSize*(core + 1), w), 0);

	for (int x = startX; x < endX; ++x)
		for (int y = 0; y < h; ++y)
			if ((x < w) && (y < h))
			{
				float sumWeights = 1;
				float sumWeightsS = 1;
				float sumWeightsD = 1;
				float currentWeights = 1;
				float currentWeightsSd = 1;
				float currentWeightsSs = 1;
				float dAlbedo = 0;
				float dNormal = 0;
				float dColor = 0;
				float dPosition = 0;
				float dDepth = 0;
				float weightS = 0;
				float weightD = 0;

				RgbaF *pPixResSs = &pix_resSpecPs[0][0] + x + y*w;
				RgbaF *pPixResDs = &pix_resDiffusePs[0][0] + x + y*w;
				RgbaF *pPixBeauty = &ResultPass.pixelsBeauty[0][0][0] + x + y*w;
				RgbaF *pPixAlbedo = &ResultPass.pixelsAlbedo[0][0][0] + x + y*w;
				RgbaF *pPixSpecular = &ResultPass.pixelsSpecular[0][0][0] + x + y*w;
				RgbaF *pPixDiffuse = &ResultPass.pixelsDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixIndirectDiffuse = &ResultPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixIndirectSpecular = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
				RgbaF *pPixRefraction = &ResultPass.pixelsRefraction[0][0][0] + x + y*w;

				Rgba *pPixTSWeight = &tSWeight[0][0] + x + y*w;
				Rgba *pPixTDWeight = &tDWeight[0][0] + x + y*w;

				RgbaF *pPixResAlbedoS = &pix_resAlbedoS[0][0] + x + y*w;
				RgbaF *pPixResSpecularS = &pix_resSpecularS[0][0] + x + y*w;
				RgbaF *pPixResDiffuseS = &pix_resDiffuseS[0][0] + x + y*w;
				RgbaF *pPixResIndDiffuseS = &pix_resIndDiffuseS[0][0] + x + y*w;
				RgbaF *pPixResIndSpecularS = &pix_resIndSpecularS[0][0] + x + y*w;
				RgbaF *pPixResRefractionS = &pix_resRefractionS[0][0] + x + y*w;

				RgbaF *pPixRes = &OutPass.pixelsBeauty[0][0][0] + x + y*w;
				RgbaF *pPixResAlbedoR = &OutPass.pixelsAlbedo[0][0][0] + x + y*w;
				RgbaF *pPixResSpecularR = &OutPass.pixelsSpecular[0][0][0] + x + y*w;
				RgbaF *pPixResDiffuseR = &OutPass.pixelsDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixResIndDiffuseR = &OutPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixResIndSpecularR = &OutPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
				RgbaF *pPixResRefractionR = &OutPass.pixelsRefraction[0][0][0] + x + y*w;

				//base
				pPixResAlbedoS->r = pPixAlbedo->r;
				pPixResAlbedoS->g = pPixAlbedo->g;
				pPixResAlbedoS->b = pPixAlbedo->b;
				pPixResAlbedoS->a = 1;

				pPixResSpecularS->r = pPixSpecular->r;
				pPixResSpecularS->g = pPixSpecular->g;
				pPixResSpecularS->b = pPixSpecular->b;
				pPixResSpecularS->a = 1;

				pPixResDiffuseS->r = pPixDiffuse->r;
				pPixResDiffuseS->g = pPixDiffuse->g;
				pPixResDiffuseS->b = pPixDiffuse->b;
				pPixResDiffuseS->a = 1;

				pPixResIndDiffuseS->r = pPixIndirectDiffuse->r;
				pPixResIndDiffuseS->g = pPixIndirectDiffuse->g;
				pPixResIndDiffuseS->b = pPixIndirectDiffuse->b;
				pPixResIndDiffuseS->a = 1;

				pPixResIndSpecularS->r = pPixIndirectSpecular->r;
				pPixResIndSpecularS->g = pPixIndirectSpecular->g;
				pPixResIndSpecularS->b = pPixIndirectSpecular->b;
				pPixResIndSpecularS->a = 1;

				pPixResRefractionS->r = pPixRefraction->r;
				pPixResRefractionS->g = pPixRefraction->g;
				pPixResRefractionS->b = pPixRefraction->b;
				pPixResRefractionS->a = 1;

				pPixResSs->r = (pPixSpecular->r + pPixIndirectSpecular->r + pPixRefraction->r);
				pPixResSs->g = (pPixSpecular->g + pPixIndirectSpecular->g + pPixRefraction->g);
				pPixResSs->b = (pPixSpecular->b + pPixIndirectSpecular->b + pPixRefraction->b);

				if (AD == 1) {
					pPixResDs->r = (pPixBeauty->r - pPixSpecular->r - pPixIndirectSpecular->r - pPixRefraction->r) / (pPixAlbedo->r + epsilon);
					pPixResDs->g = (pPixBeauty->g - pPixSpecular->g - pPixIndirectSpecular->g - pPixRefraction->g) / (pPixAlbedo->g + epsilon);
					pPixResDs->b = (pPixBeauty->b - pPixSpecular->b - pPixIndirectSpecular->b - pPixRefraction->b) / (pPixAlbedo->b + epsilon);
				}
				else
				{
					pPixResDs->r = (pPixBeauty->r - pPixSpecular->r - pPixIndirectSpecular->r - pPixRefraction->r);
					pPixResDs->g = (pPixBeauty->g - pPixSpecular->g - pPixIndirectSpecular->g - pPixRefraction->g);
					pPixResDs->b = (pPixBeauty->b - pPixSpecular->b - pPixIndirectSpecular->b - pPixRefraction->b);
				}

				RgbaF *pPixBeautyOrigA = &sigmaPass.pixelsBeauty[0][0][0] + x + y*w;
				if (pPixBeautyOrigA->a != 0) {
					int tmpCount = 0;
					sumWeights = 1;
					float sumWTS = 1;
					float sumWTD = 1;

					for (int sX = -radius; sX < radius + 1; sX++)
						for (int sY = -radius; sY < radius + 1; sY++)
							if (((x + sX) >= 0) && ((x + sX)<w))
								if (((y + sY) >= 0) && ((y + sY)<h)) {
									float ssdCd = 0;
									float ssdCs = 0;
									float ssdA = 0;
									float ssdN = 0;
									float ssdD = 0;
									float ssdP = 0;
									float ssdCdF = 0;
									float ssdCsF = 0;
									float ssdAF = 0;
									float ssdNF = 0;
									float ssdDF = 0;
									int ksize = 0;
									int frameCount = 0;
									float tmp;
									for (int frame = 1; frame < nFrames; frame++)
										if (iOpt.existsFrame[frame] == 1) {
											for (int kX = -kernel; kX < kernel + 1; kX++)
												for (int kY = -kernel; kY < kernel + 1; kY++)
													if (((x + kX) >= 0) && ((x + kX)<w))
														if (((y + kY) >= 0) && ((y + kY)<h))
															if (((x + sX + kX) >= 0) && ((x + sX + kX)<w))
																if (((y + sY + kY) >= 0) && ((y + sY + kY)<h)) {

																	RgbaF *pPixBeauty = &ResultPass.pixelsBeauty[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixBeautyKo = &ResultPass.pixelsBeauty[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	RgbaF *pPixSpecular = &ResultPass.pixelsSpecular[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixSpecularKo = &ResultPass.pixelsSpecular[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	RgbaF *pPixIndirectSpecular = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixIndirectSpecularKo = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	RgbaF *pPixRefraction = &ResultPass.pixelsRefraction[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixRefractionKo = &ResultPass.pixelsRefraction[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	tmp = ((pPixBeautyKo->r - pPixSpecularKo->r - pPixIndirectSpecularKo->r - pPixRefractionKo->r) -
																		(pPixBeauty->r - pPixSpecular->r - pPixIndirectSpecular->r - pPixRefraction->r)) +
																		((pPixBeautyKo->g - pPixSpecularKo->g - pPixIndirectSpecularKo->g - pPixRefractionKo->g) -
																		(pPixBeauty->g - pPixSpecular->g - pPixIndirectSpecular->g - pPixRefraction->g)) +
																			((pPixBeautyKo->b - pPixSpecularKo->b - pPixIndirectSpecularKo->b - pPixRefractionKo->b) -
																		(pPixBeauty->b - pPixSpecular->b - pPixIndirectSpecular->b - pPixRefraction->b));
																	if (kernel != 0) {
																		ssdCd += tmp*tmp*exp(-0.5*kX*kX / (kernel*kernel))*exp(-0.5*kY*kY / (kernel*kernel));
																	}
																	else
																	{
																		ssdCd += tmp*tmp;
																	}

																	tmp = (1 / SpecStrength)*(((pPixSpecularKo->r + pPixIndirectSpecularKo->r + pPixRefractionKo->r) -
																		(pPixSpecular->r + pPixIndirectSpecular->r + pPixRefraction->r)) +
																		((pPixSpecularKo->g + pPixIndirectSpecularKo->g + pPixRefractionKo->g) -
																		(pPixSpecular->g + pPixIndirectSpecular->g + pPixRefraction->g)) +
																			((pPixSpecularKo->b + pPixIndirectSpecularKo->b + pPixRefractionKo->b) -
																		(pPixSpecular->b + pPixIndirectSpecular->b + pPixRefraction->b)));
																	if (kernel != 0) {
																		ssdCs += tmp*tmp*exp(-0.5*kX*kX / (kernel*kernel))*exp(-0.5*kY*kY / (kernel*kernel));
																	}
																	else
																	{
																		ssdCs += tmp*tmp;
																	}
																}
											RgbaF *pPixAlbedoW = &ResultPass.pixelsAlbedo[0][0][0] + x + (y)*w;
											RgbaF *pPixAlbedoWKo = &ResultPass.pixelsAlbedo[frame][0][0] + x + sX + (y + sY)*w;

											RgbaF *pPixNormalW = &ResultPass.pixelsNormal[0][0][0] + x + (y)*w;
											RgbaF *pPixNormalWKo = &ResultPass.pixelsNormal[frame][0][0] + x + sX + (y + sY)*w;

											RgbaF *pPixDepthW = &ResultPass.pixelsDepth[0][0][0] + x + (y)*w;
											RgbaF *pPixDepthWKo = &ResultPass.pixelsDepth[frame][0][0] + x + sX + (y + sY)*w;

											RgbaF *pPixPosW = &ResultPass.pixelsPosition[0][0][0] + x + (y)*w;
											RgbaF *pPixPosWKo = &ResultPass.pixelsPosition[frame][0][0] + x + sX + (y + sY)*w;
											tmp = (pPixAlbedoWKo->r - pPixAlbedoW->r) +
												(pPixAlbedoWKo->g - pPixAlbedoW->g) +
												(pPixAlbedoWKo->b - pPixAlbedoW->b);
											ssdA += tmp*tmp;

											tmp = (pPixNormalWKo->r - pPixNormalW->r) +
												(pPixNormalWKo->g - pPixNormalW->g) +
												(pPixNormalWKo->b - pPixNormalW->b);
											ssdN += tmp*tmp;

											tmp = (pPixDepthWKo->r - pPixDepthW->r) +
												(pPixDepthWKo->g - pPixDepthW->g) +
												(pPixDepthWKo->b - pPixDepthW->b);
											ssdD += tmp*tmp;

											tmp = (pPixPosWKo->r - pPixPosW->r) +
												(pPixPosWKo->g - pPixPosW->g) +
												(pPixPosWKo->b - pPixPosW->b);
											ssdP += tmp*tmp;
											RgbaF *pPixAlbedoKorig = &ResultPass.pixelsAlbedo[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixBeautyKorig = &ResultPass.pixelsBeauty[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixSpecularKorig = &ResultPass.pixelsSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixIndirectSpecularKorig = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixRefractionKorig = &ResultPass.pixelsRefraction[frame][0][0] + x + sX + (y + sY)*w;
											Rgba *pPixFireflyWeight = &FireflyWeight[0][0] + x + y*w;
											RgbaF *pPixWeight_ = &ResultPass.pixelsBeauty[frame][0][0] + x + y*w;

											currentWeightsSd = exp(-ssdCd / (sigmaColor*sigmaColor));
											if (sigmaAlbedo != -1)
												currentWeightsSd *= exp(-ssdA / (sigmaAlbedo*sigmaAlbedo));
											if (sigmaNormal != -1)
												currentWeightsSd *= exp(-ssdN / (sigmaNormal*sigmaNormal));
											if (sigmaDepth != -1)
												currentWeightsSd *= exp(-ssdD / (sigmaDepth*sigmaDepth));
											if ((sigmaPosition != -1) && (sX == 0) && (sY == 0))
												currentWeightsSd *= exp(-ssdP / (sigmaPosition*sigmaPosition));
											currentWeightsSd *= exp(-(sqrt((float)(sX*sX + sY*sY))) / (fallof*fallof));
											currentWeightsSd *= pPixWeight_->a;

											float sigmaColorFirefly = sigmaColor;
											if ((sigmaColor < kernelOpt.ffSigma) && (firefly > 0) && (firefly <= 1))
												sigmaColorFirefly = (firefly*pPixFireflyWeight->r)*abs(sigmaColor - kernelOpt.ffSigma) + sigmaColor;
											currentWeightsSs = exp(-(ssdCs) / (sigmaColorFirefly*sigmaColorFirefly));

											if (sigmaAlbedo != -1)
												currentWeightsSs *= exp(-ssdA / (sigmaAlbedo*sigmaAlbedo));
											if (sigmaNormal != -1)
												currentWeightsSs *= exp(-ssdN / (sigmaNormal*sigmaNormal));
											if (sigmaDepth != -1)
												currentWeightsSs *= exp(-ssdD / (sigmaDepth*sigmaDepth));
											if ((sigmaPosition != -1) && (sX == 0) && (sY == 0))
												currentWeightsSs *= exp(-ssdP / (sigmaPosition*sigmaPosition));
											currentWeightsSs *= exp(-(sqrt((float)(sX*sX + sY*sY))) / (fallof*fallof));
											currentWeightsSs *= pPixWeight_->a;

											if ((sX == 0) && (sY == 0))
											{


												weightS += (pPixWeight_->a*currentWeightsSs);
												weightD += (pPixWeight_->a*currentWeightsSd);
												tmpCount++;
											}
											pPixResSs->r = pPixResSs->r + (pPixSpecularKorig->r + pPixIndirectSpecularKorig->r + pPixRefractionKorig->r)*currentWeightsSs;
											pPixResSs->g = pPixResSs->g + (pPixSpecularKorig->g + pPixIndirectSpecularKorig->g + pPixRefractionKorig->g)*currentWeightsSs;
											pPixResSs->b = pPixResSs->b + (pPixSpecularKorig->b + pPixIndirectSpecularKorig->b + pPixRefractionKorig->b)*currentWeightsSs;
											sumWeightsS += currentWeightsSs;

											RgbaF *pPixSpecK = &ResultPass.pixelsSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixDiffuseK = &ResultPass.pixelsDiffuse[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixIndDiffuseK = &ResultPass.pixelsIndirectDiffuse[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixIndSpecK = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixRefractionK = &ResultPass.pixelsRefraction[frame][0][0] + x + sX + (y + sY)*w;

											pPixResSpecularS->r = pPixResSpecularS->r + pPixSpecK->r*currentWeightsSs;
											pPixResSpecularS->g = pPixResSpecularS->g + pPixSpecK->g*currentWeightsSs;
											pPixResSpecularS->b = pPixResSpecularS->b + pPixSpecK->b*currentWeightsSs;

											pPixResIndSpecularS->r = pPixResIndSpecularS->r + pPixIndSpecK->r*currentWeightsSs;
											pPixResIndSpecularS->g = pPixResIndSpecularS->g + pPixIndSpecK->g*currentWeightsSs;
											pPixResIndSpecularS->b = pPixResIndSpecularS->b + pPixIndSpecK->b*currentWeightsSs;

											pPixResRefractionS->r = pPixResRefractionS->r + pPixRefractionK->r*currentWeightsSs;
											pPixResRefractionS->g = pPixResRefractionS->g + pPixRefractionK->g*currentWeightsSs;
											pPixResRefractionS->b = pPixResRefractionS->b + pPixRefractionK->b*currentWeightsSs;

											if (AD == 1) {
												pPixResDs->r = pPixResDs->r + (pPixBeautyKorig->r - pPixSpecularKorig->r - pPixIndirectSpecularKorig->r - pPixRefractionKorig->r) / (pPixAlbedoKorig->r + epsilon)*currentWeightsSd;
												pPixResDs->g = pPixResDs->g + (pPixBeautyKorig->g - pPixSpecularKorig->g - pPixIndirectSpecularKorig->g - pPixRefractionKorig->g) / (pPixAlbedoKorig->g + epsilon)*currentWeightsSd;
												pPixResDs->b = pPixResDs->b + (pPixBeautyKorig->b - pPixSpecularKorig->b - pPixIndirectSpecularKorig->b - pPixRefractionKorig->b) / (pPixAlbedoKorig->b + epsilon)*currentWeightsSd;
											}
											else
											{
												pPixResDs->r = pPixResDs->r + (pPixBeautyKorig->r - pPixSpecularKorig->r - pPixIndirectSpecularKorig->r - pPixRefractionKorig->r)*currentWeightsSd;
												pPixResDs->g = pPixResDs->g + (pPixBeautyKorig->g - pPixSpecularKorig->g - pPixIndirectSpecularKorig->g - pPixRefractionKorig->g)*currentWeightsSd;
												pPixResDs->b = pPixResDs->b + (pPixBeautyKorig->b - pPixSpecularKorig->b - pPixIndirectSpecularKorig->b - pPixRefractionKorig->b)*currentWeightsSd;
											}

											pPixResDiffuseS->r = pPixResDiffuseS->r + pPixDiffuseK->r*currentWeightsSd;
											pPixResDiffuseS->g = pPixResDiffuseS->g + pPixDiffuseK->g*currentWeightsSd;
											pPixResDiffuseS->b = pPixResDiffuseS->b + pPixDiffuseK->b*currentWeightsSd;

											pPixResIndDiffuseS->r = pPixResIndDiffuseS->r + pPixIndDiffuseK->r*currentWeightsSd;
											pPixResIndDiffuseS->g = pPixResIndDiffuseS->g + pPixIndDiffuseK->g*currentWeightsSd;
											pPixResIndDiffuseS->b = pPixResIndDiffuseS->b + pPixIndDiffuseK->b*currentWeightsSd;

											sumWeightsD += currentWeightsSd;
										}
								}
					if (tmpCount != 0) {
						pPixTSWeight->r = weightS / (tmpCount*1.0);
						pPixTDWeight->r = weightD / (tmpCount*1.0);
					}
					else
					{
						pPixTSWeight->r = 0;
						pPixTDWeight->r = 0;
					}
					pPixTSWeight->a = 1;
					pPixTDWeight->a = 1;
					pPixResSpecularS->r /= sumWeightsS;
					pPixResSpecularS->g /= sumWeightsS;
					pPixResSpecularS->b /= sumWeightsS;

					pPixResDiffuseS->r /= sumWeightsD;
					pPixResDiffuseS->g /= sumWeightsD;
					pPixResDiffuseS->b /= sumWeightsD;

					pPixResIndDiffuseS->r /= sumWeightsD;
					pPixResIndDiffuseS->g /= sumWeightsD;
					pPixResIndDiffuseS->b /= sumWeightsD;

					pPixResIndSpecularS->r /= sumWeightsS;
					pPixResIndSpecularS->g /= sumWeightsS;
					pPixResIndSpecularS->b /= sumWeightsS;

					pPixResRefractionS->r /= sumWeightsS;
					pPixResRefractionS->g /= sumWeightsS;
					pPixResRefractionS->b /= sumWeightsS;

					pPixResSs->r /= sumWeightsS;
					pPixResSs->g /= sumWeightsS;
					pPixResSs->b /= sumWeightsS;

					if (AD == 1) {
						pPixResDs->r /= sumWeightsD;
						pPixResDs->g /= sumWeightsD;
						pPixResDs->b /= sumWeightsD;

						pPixResDs->r *= (pPixAlbedo->r + epsilon);
						pPixResDs->g *= (pPixAlbedo->g + epsilon);
						pPixResDs->b *= (pPixAlbedo->b + epsilon);

					}
					else
					{
						pPixResDs->r /= sumWeightsD;
						pPixResDs->g /= sumWeightsD;
						pPixResDs->b /= sumWeightsD;
					}
				}
				pPixRes->r = pPixResSs->r + pPixResDs->r;
				pPixRes->g = pPixResSs->g + pPixResDs->g;
				pPixRes->b = pPixResSs->b + pPixResDs->b;
				pPixRes->a = pPixBeauty->a;

				pPixResAlbedoR->r = pPixResAlbedoS->r;
				pPixResAlbedoR->g = pPixResAlbedoS->g;
				pPixResAlbedoR->b = pPixResAlbedoS->b;
				pPixResAlbedoR->a = 1;

				pPixResSpecularR->r = pPixResSpecularS->r;
				pPixResSpecularR->g = pPixResSpecularS->g;
				pPixResSpecularR->b = pPixResSpecularS->b;
				pPixResSpecularR->a = 1;

				pPixResDiffuseR->r = pPixResDiffuseS->r;
				pPixResDiffuseR->g = pPixResDiffuseS->g;
				pPixResDiffuseR->b = pPixResDiffuseS->b;
				pPixResDiffuseR->a = 1;

				pPixResIndDiffuseR->r = pPixResIndDiffuseS->r;
				pPixResIndDiffuseR->g = pPixResIndDiffuseS->g;
				pPixResIndDiffuseR->b = pPixResIndDiffuseS->b;
				pPixResIndDiffuseR->a = 1;

				pPixResIndSpecularR->r = pPixResIndSpecularS->r;
				pPixResIndSpecularR->g = pPixResIndSpecularS->g;
				pPixResIndSpecularR->b = pPixResIndSpecularS->b;
				pPixResIndSpecularR->a = 1;

				pPixResRefractionR->r = pPixResRefractionS->r;
				pPixResRefractionR->g = pPixResRefractionS->g;
				pPixResRefractionR->b = pPixResRefractionS->b;
				pPixResRefractionR->a = 1;
			}
}

void NLM_Filter(imgPixels &sigmaPass,
	imgPixels &ResultPass,
	imgPixels &OutPass,
	float FilterWeight,
	int kernel,
	int radius,
	float sigmaColor,
	float sigmaAlbedo,
	float sigmaNormal,
	float sigmaDepth,
	float sigmaAlpha,
	float sigmaTColor,
	float sigmaTAlbedo,
	float sigmaTNormal,
	float sigmaTDepth,
	float sigmaTAlpha,
	float sigmaPosition,
	float SpecStrength,
	int nFrames,
	int AD,
	float epsilon,
	float firefly,
	imageOptions &iOpt, int core)
{
	int w = iOpt.with;
	int h = iOpt.height;
	int Pcount = 1;
	float dcount = 1;
	Array2D<RgbaF> pix_resSpecularS(w, h);
	Array2D<RgbaF> pix_resAlbedoS(w, h);
	Array2D<RgbaF> pix_resDiffuseS(w, h);
	Array2D<RgbaF> pix_resIndDiffuseS(w, h);
	Array2D<RgbaF> pix_resIndSpecularS(w, h);
	Array2D<RgbaF> pix_resRefractionS(w, h);
	Array2D<RgbaF> pix_resSpecPs(w, h);
	Array2D<RgbaF> pix_resDiffusePs(w, h);

	int CoreSize = int(w / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int x = startX; x < endX; x++)
		for (int y = 0; y < h; y++)
			if ((x < w) && (y < h))
			{
				float sumWeights = 1;
				float sumWeightsS = 1;
				float sumWeightsD = 1;
				float currentWeights = 1;
				float currentWeightsSd = 1;
				float currentWeightsSs = 1;
				float dAlbedo = 0;
				float dNormal = 0;
				float dColor = 0;
				float dPosition = 0;
				float dDepth = 0;

				RgbaF *pPixBeauty = &ResultPass.pixelsBeauty[0][0][0] + x + y*w;
				RgbaF *pPixAlbedo = &ResultPass.pixelsAlbedo[0][0][0] + x + y*w;
				RgbaF *pPixSpecular = &ResultPass.pixelsSpecular[0][0][0] + x + y*w;
				RgbaF *pPixDiffuse = &ResultPass.pixelsDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixIndirectDiffuse = &ResultPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixIndirectSpecular = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
				RgbaF *pPixRefraction = &ResultPass.pixelsRefraction[0][0][0] + x + y*w;

				RgbaF *pPixResSs = &pix_resSpecPs[0][0] + x + y*w;
				RgbaF *pPixResDs = &pix_resDiffusePs[0][0] + x + y*w;
				RgbaF *pPixResAlbedoS = &pix_resAlbedoS[0][0] + x + y*w;
				RgbaF *pPixResSpecularS = &pix_resSpecularS[0][0] + x + y*w;
				RgbaF *pPixResDiffuseS = &pix_resDiffuseS[0][0] + x + y*w;
				RgbaF *pPixResIndDiffuseS = &pix_resIndDiffuseS[0][0] + x + y*w;
				RgbaF *pPixResIndSpecularS = &pix_resIndSpecularS[0][0] + x + y*w;
				RgbaF *pPixResRefractionS = &pix_resRefractionS[0][0] + x + y*w;

				RgbaF *pPixRes = &OutPass.pixelsBeauty[0][0][0] + x + y*w;
				RgbaF *pPixResAlbedoR = &OutPass.pixelsAlbedo[0][0][0] + x + y*w;
				RgbaF *pPixResSpecularR = &OutPass.pixelsSpecular[0][0][0] + x + y*w;
				RgbaF *pPixResDiffuseR = &OutPass.pixelsDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixResIndDiffuseR = &OutPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
				RgbaF *pPixResIndSpecularR = &OutPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
				RgbaF *pPixResRefractionR = &OutPass.pixelsRefraction[0][0][0] + x + y*w;
				
				//base
				pPixResAlbedoS->r = pPixAlbedo->r;
				pPixResAlbedoS->g = pPixAlbedo->g;
				pPixResAlbedoS->b = pPixAlbedo->b;
				pPixResAlbedoS->a = 1;

				pPixResSpecularS->r = pPixSpecular->r;
				pPixResSpecularS->g = pPixSpecular->g;
				pPixResSpecularS->b = pPixSpecular->b;
				pPixResSpecularS->a = 1;

				pPixResIndSpecularS->r = pPixIndirectSpecular->r;
				pPixResIndSpecularS->g = pPixIndirectSpecular->g;
				pPixResIndSpecularS->b = pPixIndirectSpecular->b;
				pPixResIndSpecularS->a = 1;

				pPixResRefractionS->r = pPixRefraction->r;
				pPixResRefractionS->g = pPixRefraction->g;
				pPixResRefractionS->b = pPixRefraction->b;
				pPixResRefractionS->a = 1;

				pPixResSs->r = (pPixSpecular->r + pPixIndirectSpecular->r + pPixRefraction->r);
				pPixResSs->g = (pPixSpecular->g + pPixIndirectSpecular->g + pPixRefraction->g);
				pPixResSs->b = (pPixSpecular->b + pPixIndirectSpecular->b + pPixRefraction->b);

				if (AD == 1) {
					pPixResDs->r = (pPixBeauty->r - pPixSpecular->r - pPixIndirectSpecular->r - pPixRefraction->r) / (pPixAlbedo->r + epsilon);
					pPixResDs->g = (pPixBeauty->g - pPixSpecular->g - pPixIndirectSpecular->g - pPixRefraction->g) / (pPixAlbedo->g + epsilon);
					pPixResDs->b = (pPixBeauty->b - pPixSpecular->b - pPixIndirectSpecular->b - pPixRefraction->b) / (pPixAlbedo->b + epsilon);

					pPixResDiffuseS->r = pPixDiffuse->r / (pPixAlbedo->r + epsilon);
					pPixResDiffuseS->g = pPixDiffuse->g / (pPixAlbedo->g + epsilon);
					pPixResDiffuseS->b = pPixDiffuse->b / (pPixAlbedo->b + epsilon);
					pPixResDiffuseS->a = 1;

					pPixResIndDiffuseS->r = pPixIndirectDiffuse->r / (pPixAlbedo->r + epsilon);
					pPixResIndDiffuseS->g = pPixIndirectDiffuse->g / (pPixAlbedo->g + epsilon);
					pPixResIndDiffuseS->b = pPixIndirectDiffuse->b / (pPixAlbedo->b + epsilon);
					pPixResIndDiffuseS->a = 1;
				}
				else
				{
					pPixResDs->r = (pPixBeauty->r - pPixSpecular->r - pPixIndirectSpecular->r - pPixRefraction->r);
					pPixResDs->g = (pPixBeauty->g - pPixSpecular->g - pPixIndirectSpecular->g - pPixRefraction->g);
					pPixResDs->b = (pPixBeauty->b - pPixSpecular->b - pPixIndirectSpecular->b - pPixRefraction->b);

					pPixResDiffuseS->r = pPixDiffuse->r;
					pPixResDiffuseS->g = pPixDiffuse->g;
					pPixResDiffuseS->b = pPixDiffuse->b;
					pPixResDiffuseS->a = 1;

					pPixResIndDiffuseS->r = pPixIndirectDiffuse->r;
					pPixResIndDiffuseS->g = pPixIndirectDiffuse->g;
					pPixResIndDiffuseS->b = pPixIndirectDiffuse->b;
					pPixResIndDiffuseS->a = 1;
				}

				RgbaF *pPixBeautyOrigA = &sigmaPass.pixelsBeauty[0][0][0] + x + y*w;
				if (pPixBeautyOrigA->a != 0) {
					// Temporal weight
					float TemporalWT = 0;
					int tmpCount = 0;
					sumWeights = 1;
					float sumWTS = 1;
					float sumWTD = 1;
					for (int frame = 0; frame < nFrames; frame++)
						if (iOpt.existsFrame[frame] == 1)
							for (int sX = -radius; sX < radius + 1; sX++)
								for (int sY = -radius; sY < radius + 1; sY++)
									if (((x + sX) >= 0) && ((x + sX)<w))
										if (((y + sY) >= 0) && ((y + sY)<h)) {
											float ssdCd = 0;
											float ssdCs = 0;
											float ssdA = 0;
											float ssdN = 0;
											float ssdD = 0;
											float ssdP = 0;
											float ssdAlpha = 0;
											float ssdCdF = 0;
											float ssdCsF = 0;
											float ssdAF = 0;
											float ssdNF = 0;
											float ssdDF = 0;
											int ksize = 0;
											int frameCount = 0;
											float tmp;
											for (int kX = -kernel; kX < kernel + 1; kX++)
												for (int kY = -kernel; kY < kernel + 1; kY++)
													if (((x + kX) >= 0) && ((x + kX)<w))
														if (((y + kY) >= 0) && ((y + kY)<h))
															if (((x + sX + kX) >= 0) && ((x + sX + kX)<w))
																if (((y + sY + kY) >= 0) && ((y + sY + kY)<h)) {

																	RgbaF *pPixBeauty = &ResultPass.pixelsBeauty[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixBeautyKo = &ResultPass.pixelsBeauty[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	RgbaF *pPixAlbedo = &ResultPass.pixelsAlbedo[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixAlbedoKo = &ResultPass.pixelsAlbedo[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	RgbaF *pPixSpecular = &ResultPass.pixelsSpecular[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixSpecularKo = &ResultPass.pixelsSpecular[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	RgbaF *pPixIndirectSpecular = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixIndirectSpecularKo = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	RgbaF *pPixRefraction = &ResultPass.pixelsRefraction[0][0][0] + x + kX + (y + kY)*w;
																	RgbaF *pPixRefractionKo = &ResultPass.pixelsRefraction[frame][0][0] + x + sX + kX + (y + sY + kY)*w;

																	if (AD == 1) {
																		tmp = ((pPixBeautyKo->r - pPixSpecularKo->r - pPixIndirectSpecularKo->r - pPixRefractionKo->r) / (pPixAlbedoKo->r + epsilon) -
																			(pPixBeauty->r - pPixSpecular->r - pPixIndirectSpecular->r - pPixRefraction->r) / (pPixAlbedo->r + epsilon)) +
																			((pPixBeautyKo->g - pPixSpecularKo->g - pPixIndirectSpecularKo->g - pPixRefractionKo->g) / (pPixAlbedoKo->g + epsilon) -
																			(pPixBeauty->g - pPixSpecular->g - pPixIndirectSpecular->g - pPixRefraction->g) / (pPixAlbedo->g + epsilon)) +
																				((pPixBeautyKo->b - pPixSpecularKo->b - pPixIndirectSpecularKo->b - pPixRefractionKo->b) / (pPixAlbedoKo->b + epsilon) -
																			(pPixBeauty->b - pPixSpecular->b - pPixIndirectSpecular->b - pPixRefraction->b) / (pPixAlbedo->b + epsilon));
																	}
																	else
																	{
																		tmp = ((pPixBeautyKo->r - pPixSpecularKo->r - pPixIndirectSpecularKo->r - pPixRefractionKo->r) -
																			(pPixBeauty->r - pPixSpecular->r - pPixIndirectSpecular->r - pPixRefraction->r)) +
																			((pPixBeautyKo->g - pPixSpecularKo->g - pPixIndirectSpecularKo->g - pPixRefractionKo->g) -
																			(pPixBeauty->g - pPixSpecular->g - pPixIndirectSpecular->g - pPixRefraction->g)) +
																				((pPixBeautyKo->b - pPixSpecularKo->b - pPixIndirectSpecularKo->b - pPixRefractionKo->b) -
																			(pPixBeauty->b - pPixSpecular->b - pPixIndirectSpecular->b - pPixRefraction->b));
																	}

																	if (kernel != 0) {
																		ssdCd += tmp*tmp*exp(-0.5*kX*kX / (kernel*kernel))*exp(-0.5*kY*kY / (kernel*kernel));
																	}
																	else
																	{
																		ssdCd += tmp*tmp;
																	}

																	tmp = (1 / SpecStrength)*(((pPixSpecularKo->r + pPixIndirectSpecularKo->r + pPixRefractionKo->r) -
																		(pPixSpecular->r + pPixIndirectSpecular->r + pPixRefraction->r)) +
																		((pPixSpecularKo->g + pPixIndirectSpecularKo->g + pPixRefractionKo->g) -
																		(pPixSpecular->g + pPixIndirectSpecular->g + pPixRefraction->g)) +
																			((pPixSpecularKo->b + pPixIndirectSpecularKo->b + pPixRefractionKo->b) -
																		(pPixSpecular->b + pPixIndirectSpecular->b + pPixRefraction->b)));
																	if (kernel != 0) {
																		ssdCs += tmp*tmp*exp(-0.5*kX*kX / (kernel*kernel))*exp(-0.5*kY*kY / (kernel*kernel));
																	}
																	else
																	{
																		ssdCs += tmp*tmp;
																	}
																}
											RgbaF *pPixAlpha = &ResultPass.pixelsBeauty[0][0][0] + x + y*w;
											RgbaF *pPixAlphaKo = &ResultPass.pixelsBeauty[frame][0][0] + x + sX + (y + sY)*w;

											RgbaF *pPixAlbedoW = &ResultPass.pixelsAlbedo[0][0][0] + x + (y)*w;
											RgbaF *pPixAlbedoWKo = &ResultPass.pixelsAlbedo[frame][0][0] + x + sX + (y + sY)*w;

											RgbaF *pPixNormalW = &ResultPass.pixelsNormal[0][0][0] + x + (y)*w;
											RgbaF *pPixNormalWKo = &ResultPass.pixelsNormal[frame][0][0] + x + sX + (y + sY)*w;

											RgbaF *pPixDepthW = &ResultPass.pixelsDepth[0][0][0] + x + (y)*w;
											RgbaF *pPixDepthWKo = &ResultPass.pixelsDepth[frame][0][0] + x + sX + (y + sY)*w;

											RgbaF *pPixPosW = &ResultPass.pixelsPosition[0][0][0] + x + (y)*w;
											RgbaF *pPixPosWKo = &ResultPass.pixelsPosition[frame][0][0] + x + sX + (y + sY)*w;
											tmp = (pPixAlbedoWKo->r - pPixAlbedoW->r) +
												(pPixAlbedoWKo->g - pPixAlbedoW->g) +
												(pPixAlbedoWKo->b - pPixAlbedoW->b);
											ssdA += tmp*tmp;

											tmp = (pPixNormalWKo->r - pPixNormalW->r) +
												(pPixNormalWKo->g - pPixNormalW->g) +
												(pPixNormalWKo->b - pPixNormalW->b);
											ssdN += tmp*tmp;

											tmp = (pPixDepthWKo->r - pPixDepthW->r) +
												(pPixDepthWKo->g - pPixDepthW->g) +
												(pPixDepthWKo->b - pPixDepthW->b);
											ssdD += tmp*tmp;

											tmp = (pPixPosWKo->r - pPixPosW->r) +
												(pPixPosWKo->g - pPixPosW->g) +
												(pPixPosWKo->b - pPixPosW->b);
											ssdP += tmp*tmp;

											tmp = (pPixAlphaKo->a - pPixAlpha->a);
											ssdAlpha += tmp*tmp;

											RgbaF *pPixAlbedoKorig = &ResultPass.pixelsAlbedo[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixBeautyKorig = &ResultPass.pixelsBeauty[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixSpecularKorig = &ResultPass.pixelsSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixIndirectSpecularKorig = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixRefractionKorig = &ResultPass.pixelsRefraction[frame][0][0] + x + sX + (y + sY)*w;
											Rgba *pPixFireflyWeight = &FireflyWeight[0][0] + x + y*w;

											if (frame == 0)
											{
												currentWeightsSd = exp(-ssdCd / (sigmaColor*sigmaColor));
												if (sigmaAlbedo != -1)
													currentWeightsSd *= exp(-ssdA / (sigmaAlbedo*sigmaAlbedo));
												if (sigmaNormal != -1)
													currentWeightsSd *= exp(-ssdN / (sigmaNormal*sigmaNormal));
												if (sigmaDepth != -1)
													currentWeightsSd *= exp(-ssdD / (sigmaDepth*sigmaDepth));
												if (sigmaAlpha != -1)
													currentWeightsSd *= exp(-ssdAlpha / (sigmaAlpha*sigmaAlpha));
											}
											else
											{
												currentWeightsSd = exp(-ssdCd / (sigmaTColor*sigmaTColor));
												if (sigmaTAlbedo != -1)
													currentWeightsSd *= exp(-ssdA / (sigmaTAlbedo*sigmaTAlbedo));
												if (sigmaTNormal != -1)
													currentWeightsSd *= exp(-ssdN / (sigmaTNormal*sigmaTNormal));
												if (sigmaTDepth != -1)
													currentWeightsSd *= exp(-ssdD / (sigmaTDepth*sigmaTDepth));
												if (sigmaTAlpha != -1)
													currentWeightsSd *= exp(-ssdAlpha / (sigmaTAlpha*sigmaTAlpha));
											}
											if ((sigmaPosition != -1) && (frame != 0))
												currentWeightsSd *= exp(-ssdP / (sigmaPosition*sigmaPosition));

											currentWeightsSd *= FilterWeight;
											float sigmaColorFirefly;
											if (frame == 0)
											{
												sigmaColorFirefly = sigmaColor;
												if ((sigmaColor < kernelOpt.ffSigma) && (firefly > 0) && (firefly <= 1))
													sigmaColorFirefly = (firefly*pPixFireflyWeight->r)*abs(sigmaColor - kernelOpt.ffSigma) + sigmaColor;
												currentWeightsSs = exp(-(ssdCs) / (sigmaColorFirefly*sigmaColorFirefly));

												if (sigmaAlbedo != -1)
													currentWeightsSs *= exp(-ssdA / (sigmaAlbedo*sigmaAlbedo));
												if (sigmaNormal != -1)
													currentWeightsSs *= exp(-ssdN / (sigmaNormal*sigmaNormal));
												if (sigmaDepth != -1)
													currentWeightsSs *= exp(-ssdD / (sigmaDepth*sigmaDepth));
												if (sigmaAlpha != -1)
													currentWeightsSs *= exp(-ssdAlpha / (sigmaAlpha*sigmaAlpha));
											}
											else
											{
												sigmaColorFirefly = sigmaTColor;
												if ((sigmaTColor < kernelOpt.ffSigma) && (firefly > 0) && (firefly <= 1))
													sigmaColorFirefly = (firefly*pPixFireflyWeight->r)*abs(sigmaTColor - kernelOpt.ffSigma) + sigmaTColor;
												currentWeightsSs = exp(-(ssdCs) / (sigmaColorFirefly*sigmaColorFirefly));
												if (sigmaAlbedo != -1)
													currentWeightsSs *= exp(-ssdA / (sigmaTAlbedo*sigmaTAlbedo));
												if (sigmaNormal != -1)
													currentWeightsSs *= exp(-ssdN / (sigmaTNormal*sigmaTNormal));
												if (sigmaDepth != -1)
													currentWeightsSs *= exp(-ssdD / (sigmaTDepth*sigmaTDepth));
												if (sigmaAlpha != -1)
													currentWeightsSs *= exp(-ssdAlpha / (sigmaTAlpha*sigmaTAlpha));
											}
											if ((sigmaPosition != -1) && (frame != 0))
												currentWeightsSs *= exp(-ssdP / (sigmaPosition*sigmaPosition));
											currentWeightsSs *= FilterWeight;

											if ((sX != 0) || (sY != 0))
											{
												currentWeightsSs *= (1.0 - TemporalWT*(1 - 0.00));
												currentWeightsSd *= (1.0 - TemporalWT*(1 - 0.00));
											}

											pPixResSs->r = pPixResSs->r + (pPixSpecularKorig->r + pPixIndirectSpecularKorig->r + pPixRefractionKorig->r)*currentWeightsSs;
											pPixResSs->g = pPixResSs->g + (pPixSpecularKorig->g + pPixIndirectSpecularKorig->g + pPixRefractionKorig->g)*currentWeightsSs;
											pPixResSs->b = pPixResSs->b + (pPixSpecularKorig->b + pPixIndirectSpecularKorig->b + pPixRefractionKorig->b)*currentWeightsSs;
											sumWeightsS += currentWeightsSs;

											RgbaF *pPixSpecK = &ResultPass.pixelsSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixDiffuseK = &ResultPass.pixelsDiffuse[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixIndDiffuseK = &ResultPass.pixelsIndirectDiffuse[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixIndSpecK = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + sX + (y + sY)*w;
											RgbaF *pPixRefractionK = &ResultPass.pixelsRefraction[frame][0][0] + x + sX + (y + sY)*w;

											pPixResSpecularS->r = pPixResSpecularS->r + pPixSpecK->r*currentWeightsSs;
											pPixResSpecularS->g = pPixResSpecularS->g + pPixSpecK->g*currentWeightsSs;
											pPixResSpecularS->b = pPixResSpecularS->b + pPixSpecK->b*currentWeightsSs;

											pPixResIndSpecularS->r = pPixResIndSpecularS->r + pPixIndSpecK->r*currentWeightsSs;
											pPixResIndSpecularS->g = pPixResIndSpecularS->g + pPixIndSpecK->g*currentWeightsSs;
											pPixResIndSpecularS->b = pPixResIndSpecularS->b + pPixIndSpecK->b*currentWeightsSs;

											pPixResRefractionS->r = pPixResRefractionS->r + pPixRefractionK->r*currentWeightsSs;
											pPixResRefractionS->g = pPixResRefractionS->g + pPixRefractionK->g*currentWeightsSs;
											pPixResRefractionS->b = pPixResRefractionS->b + pPixRefractionK->b*currentWeightsSs;

											if (AD == 1) {
												pPixResDs->r = pPixResDs->r + ((pPixBeautyKorig->r - pPixSpecularKorig->r - pPixIndirectSpecularKorig->r - pPixRefractionKorig->r) / (pPixAlbedoKorig->r + epsilon))*currentWeightsSd;
												pPixResDs->g = pPixResDs->g + ((pPixBeautyKorig->g - pPixSpecularKorig->g - pPixIndirectSpecularKorig->g - pPixRefractionKorig->g) / (pPixAlbedoKorig->g + epsilon))*currentWeightsSd;
												pPixResDs->b = pPixResDs->b + ((pPixBeautyKorig->b - pPixSpecularKorig->b - pPixIndirectSpecularKorig->b - pPixRefractionKorig->b) / (pPixAlbedoKorig->b + epsilon))*currentWeightsSd;

												pPixResDiffuseS->r = pPixResDiffuseS->r + (pPixDiffuseK->r / (pPixAlbedoKorig->r + epsilon))*currentWeightsSd;
												pPixResDiffuseS->g = pPixResDiffuseS->g + (pPixDiffuseK->g / (pPixAlbedoKorig->g + epsilon))*currentWeightsSd;
												pPixResDiffuseS->b = pPixResDiffuseS->b + (pPixDiffuseK->b / (pPixAlbedoKorig->b + epsilon))*currentWeightsSd;

												pPixResIndDiffuseS->r = pPixResIndDiffuseS->r + (pPixIndDiffuseK->r / (pPixAlbedoKorig->r + epsilon))*currentWeightsSd;
												pPixResIndDiffuseS->g = pPixResIndDiffuseS->g + (pPixIndDiffuseK->g / (pPixAlbedoKorig->g + epsilon))*currentWeightsSd;
												pPixResIndDiffuseS->b = pPixResIndDiffuseS->b + (pPixIndDiffuseK->b / (pPixAlbedoKorig->b + epsilon))*currentWeightsSd;
											}
											else
											{
												pPixResDs->r = pPixResDs->r + (pPixBeautyKorig->r - pPixSpecularKorig->r - pPixIndirectSpecularKorig->r - pPixRefractionKorig->r)*currentWeightsSd;
												pPixResDs->g = pPixResDs->g + (pPixBeautyKorig->g - pPixSpecularKorig->g - pPixIndirectSpecularKorig->g - pPixRefractionKorig->g)*currentWeightsSd;
												pPixResDs->b = pPixResDs->b + (pPixBeautyKorig->b - pPixSpecularKorig->b - pPixIndirectSpecularKorig->b - pPixRefractionKorig->b)*currentWeightsSd;

												pPixResDiffuseS->r = pPixResDiffuseS->r + pPixDiffuseK->r*currentWeightsSd;
												pPixResDiffuseS->g = pPixResDiffuseS->g + pPixDiffuseK->g*currentWeightsSd;
												pPixResDiffuseS->b = pPixResDiffuseS->b + pPixDiffuseK->b*currentWeightsSd;

												pPixResIndDiffuseS->r = pPixResIndDiffuseS->r + pPixIndDiffuseK->r*currentWeightsSd;
												pPixResIndDiffuseS->g = pPixResIndDiffuseS->g + pPixIndDiffuseK->g*currentWeightsSd;
												pPixResIndDiffuseS->b = pPixResIndDiffuseS->b + pPixIndDiffuseK->b*currentWeightsSd;
											}
											sumWeightsD += currentWeightsSd;
										}
					if (sumWeightsS != 0) {
						pPixResSpecularS->r /= sumWeightsS;
						pPixResSpecularS->g /= sumWeightsS;
						pPixResSpecularS->b /= sumWeightsS;

						pPixResIndSpecularS->r /= sumWeightsS;
						pPixResIndSpecularS->g /= sumWeightsS;
						pPixResIndSpecularS->b /= sumWeightsS;

						pPixResRefractionS->r /= sumWeightsS;
						pPixResRefractionS->g /= sumWeightsS;
						pPixResRefractionS->b /= sumWeightsS;

						pPixResSs->r /= sumWeightsS;
						pPixResSs->g /= sumWeightsS;
						pPixResSs->b /= sumWeightsS;
					}
					RgbaF *pPixAlbedoFin = &ResultPass.pixelsAlbedo[0][0][0] + x + y*w;
					if (AD == 1) {
						if (sumWeightsD != 0) {
							pPixResDs->r /= sumWeightsD;
							pPixResDs->g /= sumWeightsD;
							pPixResDs->b /= sumWeightsD;

							pPixResDiffuseS->r /= sumWeightsD;
							pPixResDiffuseS->g /= sumWeightsD;
							pPixResDiffuseS->b /= sumWeightsD;

							pPixResIndDiffuseS->r /= sumWeightsD;
							pPixResIndDiffuseS->g /= sumWeightsD;
							pPixResIndDiffuseS->b /= sumWeightsD;
						}
						pPixResDs->r *= (pPixAlbedoFin->r + epsilon);
						pPixResDs->g *= (pPixAlbedoFin->g + epsilon);
						pPixResDs->b *= (pPixAlbedoFin->b + epsilon);

						pPixResDiffuseS->r *= (pPixAlbedoFin->r + epsilon);
						pPixResDiffuseS->g *= (pPixAlbedoFin->g + epsilon);
						pPixResDiffuseS->b *= (pPixAlbedoFin->b + epsilon);

						pPixResIndDiffuseS->r *= (pPixAlbedoFin->r + epsilon);
						pPixResIndDiffuseS->g *= (pPixAlbedoFin->g + epsilon);
						pPixResIndDiffuseS->b *= (pPixAlbedoFin->b + epsilon);
					}
					else
					{
						if (sumWeightsD != 0) {
							pPixResDs->r /= sumWeightsD;
							pPixResDs->g /= sumWeightsD;
							pPixResDs->b /= sumWeightsD;

							pPixResDiffuseS->r /= sumWeightsD;
							pPixResDiffuseS->g /= sumWeightsD;
							pPixResDiffuseS->b /= sumWeightsD;

							pPixResIndDiffuseS->r /= sumWeightsD;
							pPixResIndDiffuseS->g /= sumWeightsD;
							pPixResIndDiffuseS->b /= sumWeightsD;
						}
					}
				}
				pPixRes->r = pPixResSs->r + pPixResDs->r;
				pPixRes->g = pPixResSs->g + pPixResDs->g;
				pPixRes->b = pPixResSs->b + pPixResDs->b;
				pPixRes->a = pPixBeautyOrigA->a;

				pPixResAlbedoR->r = pPixResAlbedoS->r;
				pPixResAlbedoR->g = pPixResAlbedoS->g;
				pPixResAlbedoR->b = pPixResAlbedoS->b;
				pPixResAlbedoR->a = 1;

				pPixResSpecularR->r = pPixResSpecularS->r;
				pPixResSpecularR->g = pPixResSpecularS->g;
				pPixResSpecularR->b = pPixResSpecularS->b;
				pPixResSpecularR->a = 1;

				pPixResDiffuseR->r = pPixResDiffuseS->r;
				pPixResDiffuseR->g = pPixResDiffuseS->g;
				pPixResDiffuseR->b = pPixResDiffuseS->b;
				pPixResDiffuseR->a = 1;

				pPixResIndDiffuseR->r = pPixResIndDiffuseS->r;
				pPixResIndDiffuseR->g = pPixResIndDiffuseS->g;
				pPixResIndDiffuseR->b = pPixResIndDiffuseS->b;
				pPixResIndDiffuseR->a = 1;

				pPixResIndSpecularR->r = pPixResIndSpecularS->r;
				pPixResIndSpecularR->g = pPixResIndSpecularS->g;
				pPixResIndSpecularR->b = pPixResIndSpecularS->b;
				pPixResIndSpecularR->a = 1;

				pPixResRefractionR->r = pPixResRefractionS->r;
				pPixResRefractionR->g = pPixResRefractionS->g;
				pPixResRefractionR->b = pPixResRefractionS->b;
				pPixResRefractionR->a = 1;
			}
}

void FireflyKill_Filter(imgPixels &sigmaPass, imgPixels &ResultPass, imgPixels &OutPass, int kernel, imageOptions &iOpt, int core)
{
	int w = iOpt.with;
	int h = iOpt.height;
	int Pcount = 1;
	float dcount = 1;

	int CoreSize = int(w / iOpt.nCores) + 1;
	int startX = std::max(std::min(CoreSize*core, w), 0);
	int endX = std::max(std::min(CoreSize*(core + 1), w), 0);

	for (int x = startX; x < endX; ++x)
		for (int y = 0; y < h; ++y)
			if ((x < w - 1) && (y < h - 1))
				if ((x > 0) && (y > 0))
				{
					float sumWeights = 1;
					float sumWeightsS = 1;
					float sumWeightsD = 1;
					float currentWeights = 1;
					float currentWeightsSd = 1;
					float currentWeightsSs = 1;
					float dAlbedo = 0;
					float dNormal = 0;
					float dColor = 0;
					float dPosition = 0;
					float dDepth = 0;

					float blockMedianR[25];
					float blockMedianG[25];
					float blockMedianB[25];

					for (int frame = 0; frame < iOpt.nFrames; frame++)
						if (iOpt.existsFrame[frame] == 1) {

							RgbaF *pPixBeautyOrigA = &sigmaPass.pixelsBeauty[frame][0][0] + x + y*w;
							if (pPixBeautyOrigA->a != 0) {
								RgbaF SumPixNear;
								SumPixNear.r = 0;
								SumPixNear.g = 0;
								SumPixNear.b = 0;
								int countMedian = 0;
								for (int kX = -kernel; kX < kernel + 1; kX++)
									for (int kY = -kernel; kY < kernel + 1; kY++)
										if (((x + kX) >= 0) && ((x + kX)<w))
											if (((y + kY) >= 0) && ((y + kY)<h)) {
												RgbaF *pPixSpecular = &ResultPass.pixelsSpecular[frame][0][0] + x + kX + (y + kY)*w;
												blockMedianR[countMedian] = pPixSpecular->r;
												blockMedianG[countMedian] = pPixSpecular->g;
												blockMedianB[countMedian] = pPixSpecular->b;
												countMedian++;
											}
								std::sort(blockMedianR, blockMedianR + countMedian);
								std::sort(blockMedianG, blockMedianG + countMedian);
								std::sort(blockMedianB, blockMedianB + countMedian);

								RgbaF *pPixSpecular = &OutPass.pixelsSpecular[frame][0][0] + x + y*w;
								RgbaF *pPixBeauty = &OutPass.pixelsBeauty[frame][0][0] + x + y*w;
								RgbaF *pPixBeautyOrig = &ResultPass.pixelsBeauty[frame][0][0] + x + y*w;
								RgbaF *pPixSpecularOrig = &ResultPass.pixelsSpecular[frame][0][0] + x + y*w;
								Rgba *pPixFireflyWeight = &FireflyWeight[0][0] + x + y*w;
								float weight;
								weight = pPixFireflyWeight->r;
								int medIndex = int(countMedian / 2);
								pPixSpecular->r = blockMedianR[medIndex] * weight + pPixSpecularOrig->r*(1 - weight);
								pPixSpecular->g = blockMedianG[medIndex] * weight + pPixSpecularOrig->g*(1 - weight);
								pPixSpecular->b = blockMedianB[medIndex] * weight + pPixSpecularOrig->b*(1 - weight);

								pPixBeauty->r = pPixBeautyOrig->r - pPixSpecularOrig->r;
								pPixBeauty->g = pPixBeautyOrig->g - pPixSpecularOrig->g;
								pPixBeauty->b = pPixBeautyOrig->b - pPixSpecularOrig->b;

								pPixBeauty->r = pPixBeauty->r + pPixSpecular->r;
								pPixBeauty->g = pPixBeauty->g + pPixSpecular->g;
								pPixBeauty->b = pPixBeauty->b + pPixSpecular->b;
							}
						}
				}
}

void FireflyKill_Compute_mask(imgPixels &sigmaPass, imgPixels &ResultPass, imgPixels &OutPass, int kernel, float Gain, float Gamma, float refractionStrange, float IndirectSpecularStrange, imageOptions &iOpt, int core)
{
	int w = iOpt.with;
	int h = iOpt.height;
	int Pcount = 1;
	float dcount = 1;

	int CoreSize = int(w / iOpt.nCores) + 1;
	int startX = std::max(std::min(CoreSize*core, w), 0);
	int endX = std::max(std::min(CoreSize*(core + 1), w), 0);

	for (int x = startX; x < endX; ++x)
		for (int y = 0; y < h; ++y)
			if ((x < w) && (y < h))
				if ((x >= 0) && (y >= 0))
				{
					float sumWeights = 1;
					float sumWeightsS = 1;
					float sumWeightsD = 1;
					float currentWeights = 1;
					float currentWeightsSd = 1;
					float currentWeightsSs = 1;
					float dAlbedo = 0;
					float dNormal = 0;
					float dColor = 0;
					float dPosition = 0;
					float dDepth = 0;
					Rgba *pPixFireflyWeight = &FireflyWeight[0][0] + x + y*w;
					pPixFireflyWeight->r = 0;
					pPixFireflyWeight->g = 0;
					pPixFireflyWeight->b = 0;
					pPixFireflyWeight->a = 0;
					RgbaF *pPixBeautyOrigA = &sigmaPass.pixelsBeauty[0][0][0] + x + y*w;
					if (pPixBeautyOrigA->a != 0)
						if ((x < w) && (y < h))
							if ((x >= 0) && (y >= 0))
							{
								RgbaF SumPixNear;
								float weight = 0;
								float SumWeight = 0;
								for (int sX = -kernel; sX < kernel + 1; sX++)
									for (int sY = -kernel; sY < kernel + 1; sY++)
										if (((x + sX) >= 0) && ((x + sX)<w))
											if (((y + sY) >= 0) && ((y + sY)<h)) {
												SumPixNear.r = 0;
												SumPixNear.g = 0;
												SumPixNear.b = 0;
												int countMedian = 0;
												for (int kX = -1; kX < 2; kX++)
													for (int kY = -1; kY < 2; kY++)
														if (((x + kX + sX) >= 0) && ((x + kX + sX)<w))
															if (((y + kY + sY) >= 0) && ((y + kY + sY)<h)) {
																RgbaF *pPixSpecular = &ResultPass.pixelsSpecular[0][0][0] + x + kX + sX + (y + kY + sY)*w;
																RgbaF *pPixRefraction = &ResultPass.pixelsRefraction[0][0][0] + x + kX + sX + (y + kY + sY)*w;
																RgbaF *pPixIndirectSpecular = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + kX + sX + (y + kY + sY)*w;
																if ((kX != 0) && (kY != 0)) {
																	SumPixNear.r += pPixSpecular->r + pPixRefraction->r*refractionStrange + pPixIndirectSpecular->r*IndirectSpecularStrange;
																	SumPixNear.g += pPixSpecular->g + pPixRefraction->g*refractionStrange + pPixIndirectSpecular->g*IndirectSpecularStrange;
																	SumPixNear.b += pPixSpecular->b + pPixRefraction->b*refractionStrange + pPixIndirectSpecular->b*IndirectSpecularStrange;
																	countMedian++;
																}

															}
												SumPixNear.r /= countMedian;
												SumPixNear.g /= countMedian;
												SumPixNear.b /= countMedian;

												RgbaF *pPixSpecularOrig = &ResultPass.pixelsSpecular[0][0][0] + x + sX + (y + sY)*w;
												RgbaF *pPixRefractionOrig = &ResultPass.pixelsRefraction[0][0][0] + x + sX + (y + sY)*w;
												RgbaF *pPixIndirectSpecularOrig = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + sX + (y + sY)*w;
												weight += exp(-0.5*sX*sX / (kernel*kernel))*exp(-0.5*sY*sY / (kernel*kernel))*(abs((SumPixNear.r - pPixSpecularOrig->r + pPixRefractionOrig->r*refractionStrange + pPixIndirectSpecularOrig->r*IndirectSpecularStrange) +
													(SumPixNear.g - pPixSpecularOrig->g + pPixRefractionOrig->g*refractionStrange + pPixIndirectSpecularOrig->g*IndirectSpecularStrange) +
													(SumPixNear.b - pPixSpecularOrig->b + pPixRefractionOrig->b*refractionStrange + pPixIndirectSpecularOrig->b*IndirectSpecularStrange)) / 3);
												SumWeight += exp(-0.5*sX*sX / (2 * 2))*exp(-0.5*sY*sY / (2 * 2));
											}
								weight /= SumWeight;
								weight *= (Gain / Gamma);
								if (weight > 1)
									weight = 1;
								pPixFireflyWeight->r = weight;
								pPixFireflyWeight->a = 1;
							}
				}
}

bool isNANPixel(float r, float g, float b)
{
	bool isNAN_var = false;
	if ((r != r) || (g != g) || (b != b))
		isNAN_var = true;
	return isNAN_var;
}

void Nan_Filter(imgPixels &ResultPass, imgPixels &OutPass, imageOptions &iOpt, int core)
{
	int w = iOpt.with;
	int h = iOpt.height;
	int Pcount = 1;
	float dcount = 1;

	int CoreSize = int(w / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);

	for (int x = startX; x < endX; ++x)
		for (int y = 0; y < h; ++y)
			if ((x < w) && (y < h))
				if ((x >= 0) && (y >= 0))
				{
					for (int frame = 0; frame < iOpt.nFrames; frame++)
						if (iOpt.existsFrame[frame] == 1) {
							RgbaF *pPixBeautyOrigA = &ResultPass.pixelsBeauty[frame][0][0] + x + y*w;
							if (pPixBeautyOrigA->a != 0) {
								RgbaF *pPixRes = &OutPass.pixelsBeauty[frame][0][0] + x + y*w;
								RgbaF *pPixResAlbedo = &OutPass.pixelsAlbedo[frame][0][0] + x + y*w;
								RgbaF *pPixResSpecular = &OutPass.pixelsSpecular[frame][0][0] + x + y*w;
								RgbaF *pPixResDiffuse = &OutPass.pixelsDiffuse[frame][0][0] + x + y*w;
								RgbaF *pPixResIndDiffuse = &OutPass.pixelsIndirectDiffuse[frame][0][0] + x + y*w;
								RgbaF *pPixResIndSpecular = &OutPass.pixelsIndirectSpecular[frame][0][0] + x + y*w;
								RgbaF *pPixResRefraction = &OutPass.pixelsRefraction[frame][0][0] + x + y*w;

								RgbaF *pPixResR = &ResultPass.pixelsBeauty[frame][0][0] + x + y*w;
								RgbaF *pPixResAlbedoR = &ResultPass.pixelsAlbedo[frame][0][0] + x + y*w;
								RgbaF *pPixResSpecularR = &ResultPass.pixelsSpecular[frame][0][0] + x + y*w;
								RgbaF *pPixResDiffuseR = &ResultPass.pixelsDiffuse[frame][0][0] + x + y*w;
								RgbaF *pPixResIndDiffuseR = &ResultPass.pixelsIndirectDiffuse[frame][0][0] + x + y*w;
								RgbaF *pPixResIndSpecularR = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + y*w;
								RgbaF *pPixResRefractionR = &ResultPass.pixelsRefraction[frame][0][0] + x + y*w;

								pPixRes->r = pPixResR->r;
								pPixRes->g = pPixResR->g;
								pPixRes->b = pPixResR->b;
								pPixRes->a = pPixResR->a;
								if ((pPixResR->r != pPixResR->r) || (pPixResR->g != pPixResR->g) || (pPixResR->b != pPixResR->b)) {
									int countMerdge = 0;
									pPixRes->r = 0;
									pPixRes->g = 0;
									pPixRes->b = 0;
									for (int kX = -1; kX <= 1; kX++)
										for (int kY = -1; kY <= 1; kY++)
											if ((x + kX >= 0) && (x + kX < w))
												if ((y + kY >= 0) && (y + kY < h)) {
													RgbaF *pPixISNAN = &ResultPass.pixelsBeauty[frame][0][0] + x + kX + (y + kY)*w;
													if (isNANPixel(pPixISNAN->r, pPixISNAN->g, pPixISNAN->b) == false) {
														pPixRes->r += pPixISNAN->r;
														pPixRes->g += pPixISNAN->g;
														pPixRes->b += pPixISNAN->b;
														countMerdge++;
													}
												}

									if (countMerdge != 0) {
										pPixRes->r /= countMerdge;
										pPixRes->g /= countMerdge;
										pPixRes->b /= countMerdge;
									}
									else
									{
										pPixRes->r = 0;
										pPixRes->g = 0;
										pPixRes->b = 0;
									}
								}
								pPixResAlbedo->r = pPixResAlbedoR->r;
								pPixResAlbedo->g = pPixResAlbedoR->g;
								pPixResAlbedo->b = pPixResAlbedoR->b;
								pPixResAlbedo->a = pPixResAlbedoR->a;
								if ((pPixResAlbedoR->r != pPixResAlbedoR->r) || (pPixResAlbedoR->g != pPixResAlbedoR->g) || (pPixResAlbedoR->b != pPixResAlbedoR->b)) {
									int countMerdge = 0;
									pPixResAlbedo->r = 0;
									pPixResAlbedo->g = 0;
									pPixResAlbedo->b = 0;
									for (int kX = -1; kX <= 1; kX++)
										for (int kY = -1; kY <= 1; kY++)
											if ((x + kX >= 0) && (x + kX < w))
												if ((y + kY >= 0) && (y + kY < h)) {
													RgbaF *pPixISNAN = &ResultPass.pixelsAlbedo[frame][0][0] + x + kX + (y + kY)*w;
													if (isNANPixel(pPixISNAN->r, pPixISNAN->g, pPixISNAN->b) == false) {
														pPixResAlbedo->r += pPixISNAN->r;
														pPixResAlbedo->g += pPixISNAN->g;
														pPixResAlbedo->b += pPixISNAN->b;
														countMerdge++;
													}
												}
									if (countMerdge != 0) {
										pPixResAlbedo->r /= countMerdge;
										pPixResAlbedo->g /= countMerdge;
										pPixResAlbedo->b /= countMerdge;
									}
									else
									{
										pPixResAlbedo->r = 0;
										pPixResAlbedo->g = 0;
										pPixResAlbedo->b = 0;
									}
								}
								pPixResSpecular->r = pPixResSpecularR->r;
								pPixResSpecular->g = pPixResSpecularR->g;
								pPixResSpecular->b = pPixResSpecularR->b;
								pPixResSpecular->a = pPixResSpecularR->a;
								if ((pPixResSpecularR->r != pPixResSpecularR->r) || (pPixResSpecularR->g != pPixResSpecularR->g) || (pPixResSpecularR->b != pPixResSpecularR->b)) {
									int countMerdge = 0;
									pPixResSpecular->r = 0;
									pPixResSpecular->g = 0;
									pPixResSpecular->b = 0;
									for (int kX = -1; kX <= 1; kX++)
										for (int kY = -1; kY <= 1; kY++)
											if ((x + kX >= 0) && (x + kX < w))
												if ((y + kY >= 0) && (y + kY < h)) {
													RgbaF *pPixISNAN = &ResultPass.pixelsSpecular[frame][0][0] + x + kX + (y + kY)*w;
													if (isNANPixel(pPixISNAN->r, pPixISNAN->g, pPixISNAN->b) == false) {
														pPixResSpecular->r += pPixISNAN->r;
														pPixResSpecular->g += pPixISNAN->g;
														pPixResSpecular->b += pPixISNAN->b;
														countMerdge++;
													}
												}
									if (countMerdge != 0) {
										pPixResSpecular->r /= countMerdge;
										pPixResSpecular->g /= countMerdge;
										pPixResSpecular->b /= countMerdge;
									}
									else
									{
										pPixResSpecular->r = 0;
										pPixResSpecular->g = 0;
										pPixResSpecular->b = 0;
									}
								}
								pPixResDiffuse->r = pPixResDiffuseR->r;
								pPixResDiffuse->g = pPixResDiffuseR->g;
								pPixResDiffuse->b = pPixResDiffuseR->b;
								pPixResDiffuse->a = pPixResDiffuseR->a;
								if ((pPixResDiffuseR->r != pPixResDiffuseR->r) || (pPixResDiffuseR->g != pPixResDiffuseR->g) || (pPixResDiffuseR->b != pPixResDiffuseR->b)) {
									int countMerdge = 0;
									pPixResDiffuse->r = 0;
									pPixResDiffuse->g = 0;
									pPixResDiffuse->b = 0;
									for (int kX = -1; kX <= 1; kX++)
										for (int kY = -1; kY <= 1; kY++)
											if ((x + kX >= 0) && (x + kX < w))
												if ((y + kY >= 0) && (y + kY < h)) {
													RgbaF *pPixISNAN = &ResultPass.pixelsDiffuse[frame][0][0] + x + kX + (y + kY)*w;
													if (isNANPixel(pPixISNAN->r, pPixISNAN->g, pPixISNAN->b) == false) {
														pPixResDiffuse->r += pPixISNAN->r;
														pPixResDiffuse->g += pPixISNAN->g;
														pPixResDiffuse->b += pPixISNAN->b;
														countMerdge++;
													}
												}
									if (countMerdge != 0) {
										pPixResDiffuse->r /= countMerdge;
										pPixResDiffuse->g /= countMerdge;
										pPixResDiffuse->b /= countMerdge;
									}
									else
									{
										pPixResDiffuse->r = 0;
										pPixResDiffuse->g = 0;
										pPixResDiffuse->b = 0;
									}
								}
								pPixResIndDiffuse->r = pPixResIndDiffuseR->r;
								pPixResIndDiffuse->g = pPixResIndDiffuseR->g;
								pPixResIndDiffuse->b = pPixResIndDiffuseR->b;
								pPixResIndDiffuse->a = pPixResIndDiffuseR->a;
								if ((pPixResIndDiffuseR->r != pPixResIndDiffuseR->r) || (pPixResIndDiffuseR->g != pPixResIndDiffuseR->g) || (pPixResIndDiffuseR->b != pPixResIndDiffuseR->b)) {
									int countMerdge = 0;
									pPixResIndDiffuse->r = 0;
									pPixResIndDiffuse->g = 0;
									pPixResIndDiffuse->b = 0;
									for (int kX = -1; kX <= 1; kX++)
										for (int kY = -1; kY <= 1; kY++)
											if ((x + kX >= 0) && (x + kX < w))
												if ((y + kY >= 0) && (y + kY < h)) {
													RgbaF *pPixISNAN = &ResultPass.pixelsIndirectDiffuse[frame][0][0] + x + kX + (y + kY)*w;
													if (isNANPixel(pPixISNAN->r, pPixISNAN->g, pPixISNAN->b) == false) {
														pPixResIndDiffuse->r += pPixISNAN->r;
														pPixResIndDiffuse->g += pPixISNAN->g;
														pPixResIndDiffuse->b += pPixISNAN->b;
														countMerdge++;
													}
												}
									if (countMerdge != 0) {
										pPixResIndDiffuse->r /= countMerdge;
										pPixResIndDiffuse->g /= countMerdge;
										pPixResIndDiffuse->b /= countMerdge;
									}
									else
									{
										pPixResIndDiffuse->r = 0;
										pPixResIndDiffuse->g = 0;
										pPixResIndDiffuse->b = 0;
									}
								}
								pPixResIndSpecular->r = pPixResIndSpecularR->r;
								pPixResIndSpecular->g = pPixResIndSpecularR->g;
								pPixResIndSpecular->b = pPixResIndSpecularR->b;
								pPixResIndSpecular->a = pPixResIndSpecularR->a;
								if ((pPixResIndSpecularR->r != pPixResIndSpecularR->r) || (pPixResIndSpecularR->g != pPixResIndSpecularR->g) || (pPixResIndSpecularR->b != pPixResIndSpecularR->b)) {
									int countMerdge = 0;
									pPixResIndSpecular->r = 0;
									pPixResIndSpecular->g = 0;
									pPixResIndSpecular->b = 0;
									for (int kX = -1; kX <= 1; kX++)
										for (int kY = -1; kY <= 1; kY++)
											if ((x + kX >= 0) && (x + kX < w))
												if ((y + kY >= 0) && (y + kY < h)) {
													RgbaF *pPixISNAN = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + kX + (y + kY)*w;
													if (isNANPixel(pPixISNAN->r, pPixISNAN->g, pPixISNAN->b) == false) {
														pPixResIndSpecular->r += pPixISNAN->r;
														pPixResIndSpecular->g += pPixISNAN->g;
														pPixResIndSpecular->b += pPixISNAN->b;
														countMerdge++;
													}
												}
									if (countMerdge != 0) {
										pPixResIndSpecular->r /= countMerdge;
										pPixResIndSpecular->g /= countMerdge;
										pPixResIndSpecular->b /= countMerdge;
									}
									else
									{
										pPixResIndSpecular->r = 0;
										pPixResIndSpecular->g = 0;
										pPixResIndSpecular->b = 0;
									}
								}
								pPixResRefraction->r = pPixResRefractionR->r;
								pPixResRefraction->g = pPixResRefractionR->g;
								pPixResRefraction->b = pPixResRefractionR->b;
								pPixResRefraction->a = pPixResRefractionR->a;
								if ((pPixResRefractionR->r != pPixResRefractionR->r) || (pPixResRefractionR->g != pPixResRefractionR->g) || (pPixResRefractionR->b != pPixResRefractionR->b)) {
									int countMerdge = 0;
									pPixResRefraction->r = 0;
									pPixResRefraction->g = 0;
									pPixResRefraction->b = 0;
									for (int kX = -1; kX <= 1; kX++)
										for (int kY = -1; kY <= 1; kY++)
											if ((x + kX >= 0) && (x + kX < w))
												if ((y + kY >= 0) && (y + kY < h)) {
													RgbaF *pPixISNAN = &ResultPass.pixelsRefraction[frame][0][0] + x + kX + (y + kY)*w;
													if (isNANPixel(pPixISNAN->r, pPixISNAN->g, pPixISNAN->b) == false) {
														pPixResRefraction->r += pPixISNAN->r;
														pPixResRefraction->g += pPixISNAN->g;
														pPixResRefraction->b += pPixISNAN->b;
														countMerdge++;
													}
												}
									if (countMerdge != 0) {
										pPixResRefraction->r /= countMerdge;
										pPixResRefraction->g /= countMerdge;
										pPixResRefraction->b /= countMerdge;
									}
									else
									{
										pPixResRefraction->r = 0;
										pPixResRefraction->g = 0;
										pPixResRefraction->b = 0;
									}
								}
							}
						}
				}
}

void *NAN_Copy(imgPixels &ResultPass, imgPixels &OutPass)
{
	//thread_data_t *data = (thread_data_t *)arg;
	int w = iOpt.with;
	int h = iOpt.height;
	for (int frame = 0; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++) {

					RgbaF *pPixResR = &ResultPass.pixelsBeauty[frame][0][0] + x + y*w;
					RgbaF *pPixResAlbedoR = &ResultPass.pixelsAlbedo[frame][0][0] + x + y*w;
					RgbaF *pPixResSpecularR = &ResultPass.pixelsSpecular[frame][0][0] + x + y*w;
					RgbaF *pPixResDiffuseR = &ResultPass.pixelsDiffuse[frame][0][0] + x + y*w;
					RgbaF *pPixResIndDiffuseR = &ResultPass.pixelsIndirectDiffuse[frame][0][0] + x + y*w;
					RgbaF *pPixResIndSpecularR = &ResultPass.pixelsIndirectSpecular[frame][0][0] + x + y*w;
					RgbaF *pPixResRefractionR = &ResultPass.pixelsRefraction[frame][0][0] + x + y*w;

					RgbaF *pPixResO = &OutPass.pixelsBeauty[frame][0][0] + x + y*w;
					RgbaF *pPixResAlbedoO = &OutPass.pixelsAlbedo[frame][0][0] + x + y*w;
					RgbaF *pPixResSpecularO = &OutPass.pixelsSpecular[frame][0][0] + x + y*w;
					RgbaF *pPixResDiffuseO = &OutPass.pixelsDiffuse[frame][0][0] + x + y*w;
					RgbaF *pPixResIndDiffuseO = &OutPass.pixelsIndirectDiffuse[frame][0][0] + x + y*w;
					RgbaF *pPixResIndSpecularO = &OutPass.pixelsIndirectSpecular[frame][0][0] + x + y*w;
					RgbaF *pPixResRefractionO = &OutPass.pixelsRefraction[frame][0][0] + x + y*w;

					pPixResO->r = pPixResR->r;
					pPixResO->g = pPixResR->g;
					pPixResO->b = pPixResR->b;

					pPixResAlbedoO->r = pPixResAlbedoR->r;
					pPixResAlbedoO->g = pPixResAlbedoR->g;
					pPixResAlbedoO->b = pPixResAlbedoR->b;

					pPixResSpecularO->r = pPixResSpecularR->r;
					pPixResSpecularO->g = pPixResSpecularR->g;
					pPixResSpecularO->b = pPixResSpecularR->b;
					pPixResSpecularO->a = pPixResSpecularR->a;

					pPixResDiffuseO->r = pPixResDiffuseR->r;
					pPixResDiffuseO->g = pPixResDiffuseR->g;
					pPixResDiffuseO->b = pPixResDiffuseR->b;
					pPixResDiffuseO->a = pPixResDiffuseR->a;

					pPixResIndDiffuseO->r = pPixResIndDiffuseR->r;
					pPixResIndDiffuseO->g = pPixResIndDiffuseR->g;
					pPixResIndDiffuseO->b = pPixResIndDiffuseR->b;
					pPixResIndDiffuseO->a = pPixResIndDiffuseR->a;

					pPixResIndSpecularO->r = pPixResIndSpecularR->r;
					pPixResIndSpecularO->g = pPixResIndSpecularR->g;
					pPixResIndSpecularO->b = pPixResIndSpecularR->b;
					pPixResIndSpecularO->a = pPixResIndSpecularR->a;

					pPixResRefractionO->r = pPixResRefractionR->r;
					pPixResRefractionO->g = pPixResRefractionR->g;
					pPixResRefractionO->b = pPixResRefractionR->b;
					pPixResRefractionO->a = pPixResRefractionR->a;
				}
	return 0;
}

void cleanup(int core)
{
	int wResize, hResize;
	int wResizeB, hResizeB;
	wResize = iOpt.with;
	hResize = iOpt.height;
	wResizeB = iOpt.with;
	hResizeB = iOpt.height;

	RgbaF *pixBeauty;
	RgbaF *pixAlbedo;
	RgbaF *pixDepth;
	RgbaF *pixNormal;
	RgbaF *pixPosition;
	RgbaF *pixDiffuse;
	RgbaF *pixIndirectDiffuse;
	RgbaF *pixIndirectSpecular;
	RgbaF *pixRefraction;
	RgbaF *pixSpecular;

	Rgba *pixBeautyF;
	Rgba *pixBeautyTF;
	Rgba *pixAlbedoF;
	Rgba *pixDiffuseF;
	Rgba *pixIndirectDiffuseF;
	Rgba *pixIndirectSpecularF;
	Rgba *pixRefractionF;
	Rgba *pixSpecularF;

	Rgba *bMV;
	Rgba *bMVpos;
	RgbaF *bSmoothP;
	Rgba *tSW;
	Rgba *tDW;
	Rgba *ff;

	int div = 1;
	while (div != 0) {
		div = wResizeB % kernelOpt.tBlockSize;
		if (div != 0)
			wResizeB++;
	}

	div = 1;
	while (div != 0) {
		div = hResizeB % kernelOpt.tBlockSize;
		if (div != 0)
			hResizeB++;
	}

	int CoreSize = int((wResizeB / kernelOpt.tBlockSize) / iOpt.nCores) + 1;
	int startX = CoreSize*core;
	int endX = CoreSize*(core + 1);
	CoreSize = int(wResize / iOpt.nCores) + 1;
	startX = CoreSize*core;
	endX = CoreSize*(core + 1);

	for (int bx = startX; bx < endX; bx++)
		for (int by = 0; by < hResize; by++)
			if (bx < wResize) {
				tSW = &tSWeight[0][0] + bx + by*iOpt.with;
				tDW = &tDWeight[0][0] + bx + by*iOpt.with;
				ff = &FireflyWeight[0][0] + bx + by*iOpt.with;

				pixBeautyF = &pix_res[0][0] + bx + by*iOpt.with;
				pixBeautyTF = &pix_resT[0][0] + bx + by*iOpt.with;
				pixSpecularF = &pix_resSpecular[0][0] + bx + by*iOpt.with;
				pixAlbedoF = &pix_resAlbedo[0][0] + bx + by*iOpt.with;
				pixDiffuseF = &pix_resDiffuse[0][0] + bx + by*iOpt.with;
				pixIndirectDiffuseF = &pix_resIndDiffuse[0][0] + bx + by*iOpt.with;
				pixIndirectSpecularF = &pix_resIndSpecular[0][0] + bx + by*iOpt.with;
				pixRefractionF = &pix_resRefraction[0][0] + bx + by*iOpt.with;

				tSW->r = 0;
				tSW->g = 0;
				tSW->b = 0;
				tSW->a = 0;

				tDW->r = 0;
				tDW->g = 0;
				tDW->b = 0;
				tDW->a = 0;

				ff->r = 0;
				ff->g = 0;
				ff->b = 0;
				ff->a = 0;

				pixBeautyF->r = 0;
				pixBeautyF->g = 0;
				pixBeautyF->b = 0;
				pixBeautyF->a = 0;

				pixBeautyTF->r = 0;
				pixBeautyTF->g = 0;
				pixBeautyTF->b = 0;
				pixBeautyTF->a = 0;

				pixSpecularF->r = 0;
				pixSpecularF->g = 0;
				pixSpecularF->b = 0;
				pixSpecularF->a = 0;

				pixAlbedoF->r = 0;
				pixAlbedoF->g = 0;
				pixAlbedoF->b = 0;
				pixAlbedoF->a = 0;

				pixDiffuseF->r = 0;
				pixDiffuseF->g = 0;
				pixDiffuseF->b = 0;
				pixDiffuseF->a = 0;

				pixIndirectDiffuseF->r = 0;
				pixIndirectDiffuseF->g = 0;
				pixIndirectDiffuseF->b = 0;
				pixIndirectDiffuseF->a = 0;

				pixIndirectSpecularF->r = 0;
				pixIndirectSpecularF->g = 0;
				pixIndirectSpecularF->b = 0;
				pixIndirectSpecularF->a = 0;

				pixRefractionF->r = 0;
				pixRefractionF->g = 0;
				pixRefractionF->b = 0;
				pixRefractionF->a = 0;

				pixDepth = &ResultBlockSmoothTMP.pixelsDepth[0][0][0] + bx + by*iOpt.with;
				pixNormal = &ResultBlockSmoothTMP.pixelsNormal[0][0][0] + bx + by*iOpt.with;
				pixPosition = &ResultBlockSmoothTMP.pixelsPosition[0][0][0] + bx + by*iOpt.with;

				pixDepth->r = 0;
				pixDepth->g = 0;
				pixDepth->b = 0;
				pixDepth->a = 0;

				pixNormal->r = 0;
				pixNormal->g = 0;
				pixNormal->b = 0;
				pixNormal->a = 0;

				pixPosition->r = 0;
				pixPosition->g = 0;
				pixPosition->b = 0;
				pixPosition->a = 0;

				pixBeauty = &ResultBlockSmoothTemporalTMP.pixelsBeauty[0][0][0] + bx + by*iOpt.with;
				pixAlbedo = &ResultBlockSmoothTemporalTMP.pixelsAlbedo[0][0][0] + bx + by*iOpt.with;
				pixDepth = &ResultBlockSmoothTemporalTMP.pixelsDepth[0][0][0] + bx + by*iOpt.with;
				pixNormal = &ResultBlockSmoothTemporalTMP.pixelsNormal[0][0][0] + bx + by*iOpt.with;
				pixPosition = &ResultBlockSmoothTemporalTMP.pixelsPosition[0][0][0] + bx + by*iOpt.with;
				pixDiffuse = &ResultBlockSmoothTemporalTMP.pixelsDiffuse[0][0][0] + bx + by*iOpt.with;
				pixIndirectDiffuse = &ResultBlockSmoothTemporalTMP.pixelsIndirectDiffuse[0][0][0] + bx + by*iOpt.with;
				pixIndirectSpecular = &ResultBlockSmoothTemporalTMP.pixelsIndirectSpecular[0][0][0] + bx + by*iOpt.with;
				pixRefraction = &ResultBlockSmoothTemporalTMP.pixelsRefraction[0][0][0] + bx + by*iOpt.with;
				pixSpecular = &ResultBlockSmoothTemporalTMP.pixelsSpecular[0][0][0] + bx + by*iOpt.with;

				pixBeauty->r = 0;
				pixBeauty->g = 0;
				pixBeauty->b = 0;
				pixBeauty->a = 0;

				pixAlbedo->r = 0;
				pixAlbedo->g = 0;
				pixAlbedo->b = 0;
				pixAlbedo->a = 0;

				pixDepth->r = 0;
				pixDepth->g = 0;
				pixDepth->b = 0;
				pixDepth->a = 0;

				pixNormal->r = 0;
				pixNormal->g = 0;
				pixNormal->b = 0;
				pixNormal->a = 0;

				pixPosition->r = 0;
				pixPosition->g = 0;
				pixPosition->b = 0;
				pixPosition->a = 0;

				pixDiffuse->r = 0;
				pixDiffuse->g = 0;
				pixDiffuse->b = 0;
				pixDiffuse->a = 0;

				pixIndirectDiffuse->r = 0;
				pixIndirectDiffuse->g = 0;
				pixIndirectDiffuse->b = 0;
				pixIndirectDiffuse->a = 0;

				pixIndirectSpecular->r = 0;
				pixIndirectSpecular->g = 0;
				pixIndirectSpecular->b = 0;
				pixIndirectSpecular->a = 0;

				pixRefraction->r = 0;
				pixRefraction->g = 0;
				pixRefraction->b = 0;
				pixRefraction->a = 0;

				pixSpecular->r = 0;
				pixSpecular->g = 0;
				pixSpecular->b = 0;
				pixSpecular->a = 0;
			}

	for (int frame = 0; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int bx = startX; bx < endX; bx++)
				for (int by = 0; by < hResize; by++)
					if (bx < iOpt.with) {
						bMVpos = &blockMVPosition[frame][0][0] + bx + by*iOpt.with;
						bSmoothP = &ResultBlockSmoothPosition[frame][0][0] + bx + by*iOpt.with;

						bMVpos->r = 0;
						bMVpos->g = 0;
						bMVpos->b = 0;
						bMVpos->a = 0;

						bSmoothP->r = 0;
						bSmoothP->g = 0;
						bSmoothP->b = 0;
						bSmoothP->a = 0;

						pixBeauty = &pixels.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						pixAlbedo = &pixels.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
						pixDepth = &pixels.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
						pixNormal = &pixels.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
						pixPosition = &pixels.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
						pixDiffuse = &pixels.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectDiffuse = &pixels.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectSpecular = &pixels.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						pixRefraction = &pixels.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						pixSpecular = &pixels.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

						pixBeauty->r = 0;
						pixBeauty->g = 0;
						pixBeauty->b = 0;
						pixBeauty->a = 0;

						pixAlbedo->r = 0;
						pixAlbedo->g = 0;
						pixAlbedo->b = 0;
						pixAlbedo->a = 0;

						pixDepth->r = 0;
						pixDepth->g = 0;
						pixDepth->b = 0;
						pixDepth->a = 0;

						pixNormal->r = 0;
						pixNormal->g = 0;
						pixNormal->b = 0;
						pixNormal->a = 0;

						pixPosition->r = 0;
						pixPosition->g = 0;
						pixPosition->b = 0;
						pixPosition->a = 0;

						pixDiffuse->r = 0;
						pixDiffuse->g = 0;
						pixDiffuse->b = 0;
						pixDiffuse->a = 0;

						pixIndirectDiffuse->r = 0;
						pixIndirectDiffuse->g = 0;
						pixIndirectDiffuse->b = 0;
						pixIndirectDiffuse->a = 0;

						pixIndirectSpecular->r = 0;
						pixIndirectSpecular->g = 0;
						pixIndirectSpecular->b = 0;
						pixIndirectSpecular->a = 0;

						pixRefraction->r = 0;
						pixRefraction->g = 0;
						pixRefraction->b = 0;
						pixRefraction->a = 0;

						pixSpecular->r = 0;
						pixSpecular->g = 0;
						pixSpecular->b = 0;
						pixSpecular->a = 0;

						pixBeauty = &ResultBlockSmooth.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						pixAlbedo = &ResultBlockSmooth.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
						pixDepth = &ResultBlockSmooth.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
						pixNormal = &ResultBlockSmooth.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
						pixPosition = &ResultBlockSmooth.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
						pixDiffuse = &ResultBlockSmooth.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectDiffuse = &ResultBlockSmooth.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectSpecular = &ResultBlockSmooth.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						pixRefraction = &ResultBlockSmooth.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						pixSpecular = &ResultBlockSmooth.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

						pixBeauty->r = 0;
						pixBeauty->g = 0;
						pixBeauty->b = 0;
						pixBeauty->a = 0;

						pixAlbedo->r = 0;
						pixAlbedo->g = 0;
						pixAlbedo->b = 0;
						pixAlbedo->a = 0;

						pixDepth->r = 0;
						pixDepth->g = 0;
						pixDepth->b = 0;
						pixDepth->a = 0;

						pixNormal->r = 0;
						pixNormal->g = 0;
						pixNormal->b = 0;
						pixNormal->a = 0;

						pixPosition->r = 0;
						pixPosition->g = 0;
						pixPosition->b = 0;
						pixPosition->a = 0;

						pixDiffuse->r = 0;
						pixDiffuse->g = 0;
						pixDiffuse->b = 0;
						pixDiffuse->a = 0;

						pixIndirectDiffuse->r = 0;
						pixIndirectDiffuse->g = 0;
						pixIndirectDiffuse->b = 0;
						pixIndirectDiffuse->a = 0;

						pixIndirectSpecular->r = 0;
						pixIndirectSpecular->g = 0;
						pixIndirectSpecular->b = 0;
						pixIndirectSpecular->a = 0;

						pixRefraction->r = 0;
						pixRefraction->g = 0;
						pixRefraction->b = 0;
						pixRefraction->a = 0;

						pixSpecular->r = 0;
						pixSpecular->g = 0;
						pixSpecular->b = 0;
						pixSpecular->a = 0;

						pixBeauty = &ResultBlockSmoothTemporal.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						pixAlbedo = &ResultBlockSmoothTemporal.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
						pixDepth = &ResultBlockSmoothTemporal.pixelsDepth[frame][0][0] + bx + by*iOpt.with;
						pixNormal = &ResultBlockSmoothTemporal.pixelsNormal[frame][0][0] + bx + by*iOpt.with;
						pixPosition = &ResultBlockSmoothTemporal.pixelsPosition[frame][0][0] + bx + by*iOpt.with;
						pixDiffuse = &ResultBlockSmoothTemporal.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectDiffuse = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectSpecular = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						pixRefraction = &ResultBlockSmoothTemporal.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						pixSpecular = &ResultBlockSmoothTemporal.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

						pixBeauty->r = 0;
						pixBeauty->g = 0;
						pixBeauty->b = 0;
						pixBeauty->a = 0;

						pixAlbedo->r = 0;
						pixAlbedo->g = 0;
						pixAlbedo->b = 0;
						pixAlbedo->a = 0;

						pixDepth->r = 0;
						pixDepth->g = 0;
						pixDepth->b = 0;
						pixDepth->a = 0;

						pixNormal->r = 0;
						pixNormal->g = 0;
						pixNormal->b = 0;
						pixNormal->a = 0;

						pixPosition->r = 0;
						pixPosition->g = 0;
						pixPosition->b = 0;
						pixPosition->a = 0;

						pixDiffuse->r = 0;
						pixDiffuse->g = 0;
						pixDiffuse->b = 0;
						pixDiffuse->a = 0;

						pixIndirectDiffuse->r = 0;
						pixIndirectDiffuse->g = 0;
						pixIndirectDiffuse->b = 0;
						pixIndirectDiffuse->a = 0;

						pixIndirectSpecular->r = 0;
						pixIndirectSpecular->g = 0;
						pixIndirectSpecular->b = 0;
						pixIndirectSpecular->a = 0;

						pixRefraction->r = 0;
						pixRefraction->g = 0;
						pixRefraction->b = 0;
						pixRefraction->a = 0;

						pixSpecular->r = 0;
						pixSpecular->g = 0;
						pixSpecular->b = 0;
						pixSpecular->a = 0;

						pixBeauty = &ResultBlockSmoothTMP.pixelsBeauty[frame][0][0] + bx + by*iOpt.with;
						pixAlbedo = &ResultBlockSmoothTMP.pixelsAlbedo[frame][0][0] + bx + by*iOpt.with;
						pixDiffuse = &ResultBlockSmoothTMP.pixelsDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectDiffuse = &ResultBlockSmoothTMP.pixelsIndirectDiffuse[frame][0][0] + bx + by*iOpt.with;
						pixIndirectSpecular = &ResultBlockSmoothTMP.pixelsIndirectSpecular[frame][0][0] + bx + by*iOpt.with;
						pixRefraction = &ResultBlockSmoothTMP.pixelsRefraction[frame][0][0] + bx + by*iOpt.with;
						pixSpecular = &ResultBlockSmoothTMP.pixelsSpecular[frame][0][0] + bx + by*iOpt.with;

						pixBeauty->r = 0;
						pixBeauty->g = 0;
						pixBeauty->b = 0;
						pixBeauty->a = 0;

						pixAlbedo->r = 0;
						pixAlbedo->g = 0;
						pixAlbedo->b = 0;
						pixAlbedo->a = 0;

						pixDiffuse->r = 0;
						pixDiffuse->g = 0;
						pixDiffuse->b = 0;
						pixDiffuse->a = 0;

						pixIndirectDiffuse->r = 0;
						pixIndirectDiffuse->g = 0;
						pixIndirectDiffuse->b = 0;
						pixIndirectDiffuse->a = 0;

						pixIndirectSpecular->r = 0;
						pixIndirectSpecular->g = 0;
						pixIndirectSpecular->b = 0;
						pixIndirectSpecular->a = 0;

						pixRefraction->r = 0;
						pixRefraction->g = 0;
						pixRefraction->b = 0;
						pixRefraction->a = 0;

						pixSpecular->r = 0;
						pixSpecular->g = 0;
						pixSpecular->b = 0;
						pixSpecular->a = 0;
					}
}

void *CleanupMemory(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	cleanup(data->core);
	return 0;
}

void *UnpremultImages(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	unpremult(data->core);
	return 0;
}

void *MotionCompensation(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;

	if ((kernelOpt.kernelMode == "ST") || (kernelOpt.kernelMode == "STPW") || (kernelOpt.kernelMode == "S")) {
		BlockSmoothSpatial(data->core);
	}

	if ((kernelOpt.kernelMode == "STPW") || (kernelOpt.kernelMode == "TPW")) {
		searchBlockDiamond4StepPW(data->core);
	}

	if ((kernelOpt.kernelMode == "ST") || (kernelOpt.kernelMode == "T")) {
		searchBlockDiamond4Step(data->core);
	}
	return 0;
}

void *BlockMatch(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	if ((kernelOpt.kernelMode == "STPW") || (kernelOpt.kernelMode == "TPW")) {
		BlockSmoothPW2(data->core);
	}
	if ((kernelOpt.kernelMode == "ST") || (kernelOpt.kernelMode == "T")) {
		BlockSmooth(data->core);
	}
	return 0;
}

void BlockMatchMV()
{
	if ((kernelOpt.kernelMode == "ST") || (kernelOpt.kernelMode == "T")) {
		BlockMVSmooth();
	}
}

void *NAN_check(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	Nan_Filter(pixels, ResultBlockSmoothTMP, iOpt, data->core);
	return 0;
}

void *Firefly_Kill(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	FireflyKill_Compute_mask(pixels, pixels,
		ResultBlockSmoothTMP, 
		kernelOpt.ffkernel, 
		kernelOpt.ffGain, 
		kernelOpt.ffGamma, 
		kernelOpt.ffRefractionStrange, 
		kernelOpt.ffindirectSpecStrange, 
		iOpt, data->core);
	return 0;
}

void *NLM_First(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	if ((kernelOpt.kernelMode == "STPW") || (kernelOpt.kernelMode == "ST") || (kernelOpt.kernelMode == "S")) {
		NLM_Filter(pixels, ResultBlockSmooth,
			ResultBlockSmoothTMP,
			kernelOpt.sWeight,
			kernelOpt.skernel,
			kernelOpt.sradius,
			kernelOpt.sColor,
			kernelOpt.sAlbedo,
			kernelOpt.sNormal,
			kernelOpt.sDepth,
			kernelOpt.sAlpha,
			kernelOpt.stColor,
			kernelOpt.stAlbedo,
			kernelOpt.stNormal,
			kernelOpt.stDepth,
			kernelOpt.stAlpha,
			-1,
			kernelOpt.sSpecularStrength,
			iOpt.nFrames,
			kernelOpt.albedoDivide,
			kernelOpt.albedoTreshold,
			1,
			iOpt, data->core);
	}
	return 0;
}

void *NLM_FinalTouch(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	if ((kernelOpt.kernelMode == "STPW") || (kernelOpt.kernelMode == "ST") || (kernelOpt.kernelMode == "S")) {
		NLM_Filter(pixels, ResultBlockSmooth,
			ResultBlockSmoothTMP,
			kernelOpt.sfWeight,
			kernelOpt.sfkernel,
			kernelOpt.sfradius,
			kernelOpt.sfColor,
			kernelOpt.sfAlbedo,
			kernelOpt.sfNormal,
			kernelOpt.sfDepth,
			kernelOpt.sAlpha,
			kernelOpt.stColor,
			kernelOpt.stAlbedo,
			kernelOpt.stNormal,
			kernelOpt.stDepth,
			kernelOpt.stAlpha,
			-1,
			kernelOpt.sfSpecularStrength,
			1,
			kernelOpt.sfAlbedoDivide,
			kernelOpt.albedoTreshold,
			0,
			iOpt, data->core);
	}
	return 0;
}

void *NLM_Temporal(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;
	if ((kernelOpt.kernelMode == "STPW") || (kernelOpt.kernelMode == "TPW")) {
		NLM_Temporal_Filter(pixels, ResultBlockSmoothTemporal,
			ResultBlockSmoothTemporalTMP,
			kernelOpt.tPwPRadius,
			kernelOpt.tPwKernel,
			kernelOpt.tPwSigmaColor,
			kernelOpt.tPwSigmaAlbedo,
			-1,
			-1,
			kernelOpt.tPwSigmaDistance,
			kernelOpt.sfSpecularStrength,
			iOpt.nFrames,
			0,
			kernelOpt.albedoTreshold,
			1,
			kernelOpt.tPwFallof,
			iOpt, data->core);
	}
	if ((kernelOpt.kernelMode == "ST") || (kernelOpt.kernelMode == "T")) {
		NLM_Temporal_Filter(pixels, ResultBlockSmoothTemporal,
			ResultBlockSmoothTemporalTMP,
			kernelOpt.tPRadius,
			kernelOpt.tkernel,
			kernelOpt.tSigmaColor,
			kernelOpt.tSigmaAlbedo,
			-1,
			-1,
			-1,
			kernelOpt.sfSpecularStrength,
			iOpt.nFrames,
			0,
			kernelOpt.albedoTreshold,
			1,
			kernelOpt.tFallof,
			iOpt, data->core);
	}
	return 0;
}

void *Filter_merge(void *arg)
{
	thread_data_t *data = (thread_data_t *)arg;

	int w = iOpt.with;
	int h = iOpt.height;

	int CoreSize = int(w / iOpt.nCores) + 1;
	int startX = std::max(std::min(CoreSize*data->core, w), 0);
	int endX = std::max(std::min(CoreSize*(data->core + 1), w), 0);
	for (int x = startX; x < endX; x++)
		for (int y = 0; y < h; y++) {

			Rgba *pPixTSWeight = &tSWeight[0][0] + x + y*w;
			Rgba *pPixTDWeight = &tDWeight[0][0] + x + y*w;

			Rgba *pPixRes = &pix_res[0][0] + x + y*w;
			Rgba *pPixResAlbedo = &pix_resAlbedo[0][0] + x + y*w;
			Rgba *pPixResSpecular = &pix_resSpecular[0][0] + x + y*w;
			Rgba *pPixResDiffuse = &pix_resDiffuse[0][0] + x + y*w;
			Rgba *pPixResIndDiffuse = &pix_resIndDiffuse[0][0] + x + y*w;
			Rgba *pPixResIndSpecular = &pix_resIndSpecular[0][0] + x + y*w;
			Rgba *pPixResRefraction = &pix_resRefraction[0][0] + x + y*w;

			RgbaF *pPixResF = &ResultBlockSmooth.pixelsBeauty[0][0][0] + x + y*w;
			RgbaF pPixResSF;
			RgbaF pPixResDF;
			RgbaF *pPixResAlbedoF = &ResultBlockSmooth.pixelsAlbedo[0][0][0] + x + y*w;
			RgbaF *pPixResSpecularF = &ResultBlockSmooth.pixelsSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResDiffuseF = &ResultBlockSmooth.pixelsDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndDiffuseF = &ResultBlockSmooth.pixelsIndirectDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndSpecularF = &ResultBlockSmooth.pixelsIndirectSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResRefractionF = &ResultBlockSmooth.pixelsRefraction[0][0][0] + x + y*w;

			RgbaF *pPixResFT = &ResultBlockSmoothTemporal.pixelsBeauty[0][0][0] + x + y*w;
			RgbaF pPixResSFT;
			RgbaF pPixResDFT;
			RgbaF *pPixResAlbedoFT = &ResultBlockSmoothTemporal.pixelsAlbedo[0][0][0] + x + y*w;
			RgbaF *pPixResSpecularFT = &ResultBlockSmoothTemporal.pixelsSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResDiffuseFT = &ResultBlockSmoothTemporal.pixelsDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndDiffuseFT = &ResultBlockSmoothTemporal.pixelsIndirectDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndSpecularFT = &ResultBlockSmoothTemporal.pixelsIndirectSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResRefractionFT = &ResultBlockSmoothTemporal.pixelsRefraction[0][0][0] + x + y*w;

			float weightS = pPixTSWeight->r;
			float weightD = pPixTDWeight->r;

			pPixResSFT.r = pPixResSpecularFT->r + pPixResIndSpecularFT->r + pPixResRefractionFT->r;
			pPixResSFT.g = pPixResSpecularFT->g + pPixResIndSpecularFT->g + pPixResRefractionFT->g;
			pPixResSFT.b = pPixResSpecularFT->b + pPixResIndSpecularFT->b + pPixResRefractionFT->b;
			pPixResSFT.a = pPixResFT->a;

			pPixResDFT.r = pPixResFT->r - pPixResSpecularFT->r - pPixResIndSpecularFT->r - pPixResRefractionFT->r;
			pPixResDFT.g = pPixResFT->g - pPixResSpecularFT->g - pPixResIndSpecularFT->g - pPixResRefractionFT->g;
			pPixResDFT.b = pPixResFT->b - pPixResSpecularFT->b - pPixResIndSpecularFT->b - pPixResRefractionFT->b;
			pPixResDFT.a = pPixResFT->a;

			pPixResSF.r = pPixResSpecularF->r + pPixResIndSpecularF->r + pPixResRefractionF->r;
			pPixResSF.g = pPixResSpecularF->g + pPixResIndSpecularF->g + pPixResRefractionF->g;
			pPixResSF.b = pPixResSpecularF->b + pPixResIndSpecularF->b + pPixResRefractionF->b;
			pPixResSF.a = pPixResF->a;

			pPixResDF.r = pPixResF->r - pPixResSpecularF->r - pPixResIndSpecularF->r - pPixResRefractionF->r;
			pPixResDF.g = pPixResF->g - pPixResSpecularF->g - pPixResIndSpecularF->g - pPixResRefractionF->g;
			pPixResDF.b = pPixResF->b - pPixResSpecularF->b - pPixResIndSpecularF->b - pPixResRefractionF->b;
			pPixResDF.a = pPixResF->a;

			pPixRes->r = pPixResSF.r*(1 - weightS) + pPixResDF.r*(1 - weightD) +
				pPixResSFT.r*weightS + pPixResDFT.r*weightD;
			pPixRes->g = pPixResSF.g*(1 - weightS) + pPixResDF.g*(1 - weightD) +
				pPixResSFT.g*weightS + pPixResDFT.g*weightD;
			pPixRes->b = pPixResSF.b*(1 - weightS) + pPixResDF.b*(1 - weightD) +
				pPixResSFT.b*weightS + pPixResDFT.b*weightD;
			pPixRes->a = pPixResF->a;

			pPixResAlbedo->r = pPixResAlbedoF->r*(1 - weightD) + pPixResAlbedoFT->r*weightD;
			pPixResAlbedo->g = pPixResAlbedoF->g*(1 - weightD) + pPixResAlbedoFT->g*weightD;
			pPixResAlbedo->b = pPixResAlbedoF->b*(1 - weightD) + pPixResAlbedoFT->b*weightD;
			pPixResAlbedo->a = 1;

			pPixResSpecular->r = pPixResSpecularF->r*(1 - weightS) + pPixResSpecularFT->r*weightS;
			pPixResSpecular->g = pPixResSpecularF->g*(1 - weightS) + pPixResSpecularFT->g*weightS;
			pPixResSpecular->b = pPixResSpecularF->b*(1 - weightS) + pPixResSpecularFT->b*weightS;
			pPixResSpecular->a = 1;

			pPixResDiffuse->r = pPixResDiffuseF->r*(1 - weightD) + pPixResDiffuseFT->r*weightD;
			pPixResDiffuse->g = pPixResDiffuseF->g*(1 - weightD) + pPixResDiffuseFT->g*weightD;
			pPixResDiffuse->b = pPixResDiffuseF->b*(1 - weightD) + pPixResDiffuseFT->b*weightD;
			pPixResDiffuse->a = 1;

			pPixResIndDiffuse->r = pPixResIndDiffuseF->r*(1 - weightD) + pPixResIndDiffuseFT->r*weightD;
			pPixResIndDiffuse->g = pPixResIndDiffuseF->g*(1 - weightD) + pPixResIndDiffuseFT->g*weightD;
			pPixResIndDiffuse->b = pPixResIndDiffuseF->b*(1 - weightD) + pPixResIndDiffuseFT->b*weightD;
			pPixResIndDiffuse->a = 1;

			pPixResIndSpecular->r = pPixResIndSpecularF->r*(1 - weightS) + pPixResIndSpecularFT->r*weightS;
			pPixResIndSpecular->g = pPixResIndSpecularF->g*(1 - weightS) + pPixResIndSpecularFT->g*weightS;
			pPixResIndSpecular->b = pPixResIndSpecularF->b*(1 - weightS) + pPixResIndSpecularFT->b*weightS;
			pPixResIndSpecular->a = 1;

			pPixResRefraction->r = pPixResRefractionF->r*(1 - weightS) + pPixResRefractionFT->r*weightS;
			pPixResRefraction->g = pPixResRefractionF->g*(1 - weightS) + pPixResRefractionFT->g*weightS;
			pPixResRefraction->b = pPixResRefractionF->b*(1 - weightS) + pPixResRefractionFT->b*weightS;
			pPixResRefraction->a = 1;

			// premult
			pPixRes->r *= pPixResF->a;
			pPixRes->g *= pPixResF->a;
			pPixRes->b *= pPixResF->a;
			pPixRes->a = pPixResF->a;

			pPixResAlbedo->r *= pPixResF->a;
			pPixResAlbedo->g *= pPixResF->a;
			pPixResAlbedo->b *= pPixResF->a;

			pPixResSpecular->r *= pPixResF->a;
			pPixResSpecular->g *= pPixResF->a;
			pPixResSpecular->b *= pPixResF->a;

			pPixResDiffuse->r *= pPixResF->a;
			pPixResDiffuse->g *= pPixResF->a;
			pPixResDiffuse->b *= pPixResF->a;

			pPixResIndDiffuse->r *= pPixResF->a;
			pPixResIndDiffuse->g *= pPixResF->a;
			pPixResIndDiffuse->b *= pPixResF->a;

			pPixResIndSpecular->r *= pPixResF->a;
			pPixResIndSpecular->g *= pPixResF->a;
			pPixResIndSpecular->b *= pPixResF->a;

			pPixResRefraction->r *= pPixResF->a;
			pPixResRefraction->g *= pPixResF->a;
			pPixResRefraction->b *= pPixResF->a;

			//NAN
			if (pPixResF->a == 0) {
				pPixResAlbedo->r = 0;
				pPixResAlbedo->g = 0;
				pPixResAlbedo->b = 0;

				pPixResSpecular->r = 0;
				pPixResSpecular->g = 0;
				pPixResSpecular->b = 0;

				pPixResDiffuse->r = 0;
				pPixResDiffuse->g = 0;
				pPixResDiffuse->b = 0;

				pPixResIndDiffuse->r = 0;
				pPixResIndDiffuse->g = 0;
				pPixResIndDiffuse->b = 0;

				pPixResIndSpecular->r = 0;
				pPixResIndSpecular->g = 0;
				pPixResIndSpecular->b = 0;

				pPixResRefraction->r = 0;
				pPixResRefraction->g = 0;
				pPixResRefraction->b = 0;
			}
		}
	return 0;
}

void *TMP_Copy(imgPixels &ResultPass, imgPixels &OutPass)
{
	int w = iOpt.with;
	int h = iOpt.height;

	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++) {

			RgbaF *pPixResR = &ResultPass.pixelsBeauty[0][0][0] + x + y*w;
			RgbaF *pPixResAlbedoR = &ResultPass.pixelsAlbedo[0][0][0] + x + y*w;
			RgbaF *pPixResSpecularR = &ResultPass.pixelsSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResDiffuseR = &ResultPass.pixelsDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndDiffuseR = &ResultPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndSpecularR = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResRefractionR = &ResultPass.pixelsRefraction[0][0][0] + x + y*w;

			RgbaF *pPixResO = &OutPass.pixelsBeauty[0][0][0] + x + y*w;
			RgbaF *pPixResAlbedoO = &OutPass.pixelsAlbedo[0][0][0] + x + y*w;
			RgbaF *pPixResSpecularO = &OutPass.pixelsSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResDiffuseO = &OutPass.pixelsDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndDiffuseO = &OutPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
			RgbaF *pPixResIndSpecularO = &OutPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
			RgbaF *pPixResRefractionO = &OutPass.pixelsRefraction[0][0][0] + x + y*w;

			pPixResO->r = pPixResR->r;
			pPixResO->g = pPixResR->g;
			pPixResO->b = pPixResR->b;
			pPixResO->a = pPixResR->a;

			pPixResAlbedoO->r = pPixResAlbedoR->r;
			pPixResAlbedoO->g = pPixResAlbedoR->g;
			pPixResAlbedoO->b = pPixResAlbedoR->b;
			pPixResAlbedoO->a = pPixResAlbedoR->a;

			pPixResSpecularO->r = pPixResSpecularR->r;
			pPixResSpecularO->g = pPixResSpecularR->g;
			pPixResSpecularO->b = pPixResSpecularR->b;
			pPixResSpecularO->a = pPixResSpecularR->a;

			pPixResDiffuseO->r = pPixResDiffuseR->r;
			pPixResDiffuseO->g = pPixResDiffuseR->g;
			pPixResDiffuseO->b = pPixResDiffuseR->b;
			pPixResDiffuseO->a = pPixResDiffuseR->a;

			pPixResIndDiffuseO->r = pPixResIndDiffuseR->r;
			pPixResIndDiffuseO->g = pPixResIndDiffuseR->g;
			pPixResIndDiffuseO->b = pPixResIndDiffuseR->b;
			pPixResIndDiffuseO->a = pPixResIndDiffuseR->a;

			pPixResIndSpecularO->r = pPixResIndSpecularR->r;
			pPixResIndSpecularO->g = pPixResIndSpecularR->g;
			pPixResIndSpecularO->b = pPixResIndSpecularR->b;
			pPixResIndSpecularO->a = pPixResIndSpecularR->a;

			pPixResRefractionO->r = pPixResRefractionR->r;
			pPixResRefractionO->g = pPixResRefractionR->g;
			pPixResRefractionO->b = pPixResRefractionR->b;
			pPixResRefractionO->a = pPixResRefractionR->a;
		}
	return 0;
}

void *TMP_Firefly_Copy(imgPixels &ResultPass, imgPixels &OutPass)
{
	//thread_data_t *data = (thread_data_t *)arg;
	int w = iOpt.with;
	int h = iOpt.height;
	for (int frame = 0; frame < iOpt.nFrames; frame++)
		if (iOpt.existsFrame[frame] == 1)
			for (int x = 0; x < w; x++)
				for (int y = 0; y < h; y++) {
					RgbaF *pPixResR = &ResultPass.pixelsBeauty[frame][0][0] + x + y*w;
					RgbaF *pPixResAlbedoR = &ResultPass.pixelsAlbedo[0][0][0] + x + y*w;
					RgbaF *pPixResSpecularR = &ResultPass.pixelsSpecular[frame][0][0] + x + y*w;
					RgbaF *pPixResDiffuseR = &ResultPass.pixelsDiffuse[0][0][0] + x + y*w;
					RgbaF *pPixResIndDiffuseR = &ResultPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
					RgbaF *pPixResIndSpecularR = &ResultPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
					RgbaF *pPixResRefractionR = &ResultPass.pixelsRefraction[0][0][0] + x + y*w;

					RgbaF *pPixResO = &OutPass.pixelsBeauty[frame][0][0] + x + y*w;
					RgbaF *pPixResAlbedoO = &OutPass.pixelsAlbedo[0][0][0] + x + y*w;
					RgbaF *pPixResSpecularO = &OutPass.pixelsSpecular[frame][0][0] + x + y*w;
					RgbaF *pPixResDiffuseO = &OutPass.pixelsDiffuse[0][0][0] + x + y*w;
					RgbaF *pPixResIndDiffuseO = &OutPass.pixelsIndirectDiffuse[0][0][0] + x + y*w;
					RgbaF *pPixResIndSpecularO = &OutPass.pixelsIndirectSpecular[0][0][0] + x + y*w;
					RgbaF *pPixResRefractionO = &OutPass.pixelsRefraction[0][0][0] + x + y*w;

					pPixResO->r = pPixResR->r;
					pPixResO->g = pPixResR->g;
					pPixResO->b = pPixResR->b;

					pPixResSpecularO->r = pPixResSpecularR->r;
					pPixResSpecularO->g = pPixResSpecularR->g;
					pPixResSpecularO->b = pPixResSpecularR->b;
				}
	return 0;
}

void WritePasses(int frame)
{
	// write beauty pass
	string str = iOpt.fnameBeauty;
	string str2("_variance.");
	str.replace(str.find(str2), str2.length(), ".");

	std::string filePath = iOpt.fnameBeauty;
	filePath = str;
	filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), iOpt.oPostfix);
	writeRgba(filePath.c_str(), &pix_res[0][0], iOpt.with, iOpt.height);
	
	// write diffuse pass
	if (iOpt.fnameDiffuse != "") {
		filePath = iOpt.fnameDiffuse;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), iOpt.oPostfix);
		writeRgba(filePath.c_str(), &pix_resDiffuse[0][0], iOpt.with, iOpt.height);
	}
	
	// write specular pass
	if (iOpt.fnameSpecular != "") {
		filePath = iOpt.fnameSpecular;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), iOpt.oPostfix);
		writeRgba(filePath.c_str(), &pix_resSpecular[0][0], iOpt.with, iOpt.height);
	}
	// write indirectdiffuse pass
	if (iOpt.fnameIndirectDiffuse != "") {
		filePath = iOpt.fnameIndirectDiffuse;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), iOpt.oPostfix);
		writeRgba(filePath.c_str(), &pix_resIndDiffuse[0][0], iOpt.with, iOpt.height);
	}

	// write indirectspecular pass
	if (iOpt.fnameIndirectSpecular != "") {
		filePath = iOpt.fnameIndirectSpecular;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), iOpt.oPostfix);
		writeRgba(filePath.c_str(), &pix_resIndSpecular[0][0], iOpt.with, iOpt.height);
	}

	// write albedo
	if (iOpt.fnameAlbedo != "") {
		filePath = iOpt.fnameAlbedo;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), iOpt.oPostfix);
		writeRgba(filePath.c_str(), &pix_resAlbedo[0][0], iOpt.with, iOpt.height);
	}

	// write refraction
	if (iOpt.fnameRefraction != "") {
		filePath = iOpt.fnameRefraction;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), iOpt.oPostfix);
		writeRgba(filePath.c_str(), &pix_resRefraction[0][0], iOpt.with, iOpt.height);
	}

	if (iOpt.fnameBeauty != "") {
		filePath = iOpt.fnameBeauty;
		filePath = str;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), "_wtMapS");
		writeRgba(filePath.c_str(), &tSWeight[0][0], iOpt.with, iOpt.height);
	}

	if (iOpt.fnameBeauty != "") {
		filePath = iOpt.fnameBeauty;
		filePath = str;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), "_wtMapD");
		writeRgba(filePath.c_str(), &tDWeight[0][0], iOpt.with, iOpt.height);
	}

	if (iOpt.fnameBeauty != "") {
		filePath = iOpt.fnameBeauty;
		filePath = str;
		filePath = OutFileName(filePath, frame, getPadding(iOpt.fnameBeauty), "_FireflyMask");
		writeRgba(filePath.c_str(), &FireflyWeight[0][0], iOpt.with, iOpt.height);
	}
}

void PrintFilterOpt()
{
	cout << "" << endl;
	cout << "FILTER SETTINGS" << endl;
	cout << "nCores " << iOpt.nCores << endl;
	cout << "nFrame " << iOpt.nFrames << endl;
	cout << "kernel mode " << kernelOpt.kernelMode << endl;

	//kernel spatial
	cout << "skernel " << kernelOpt.skernel << endl;
	cout << "sradius " << kernelOpt.sradius << endl;
	cout << "sWeight " << kernelOpt.sWeight << endl;
	cout << "sColor " << kernelOpt.sColor << endl;
	cout << "sAlbedo " << kernelOpt.sAlbedo << endl;
	cout << "sNormal " << kernelOpt.sNormal << endl;
	cout << "sDepth " << kernelOpt.sDepth << endl;
	cout << "sAlpha " << kernelOpt.sAlpha << endl;
	cout << "stColor " << kernelOpt.stColor << endl;
	cout << "stAlbedo " << kernelOpt.stAlbedo << endl;
	cout << "stNormal " << kernelOpt.stNormal << endl;
	cout << "stDepth " << kernelOpt.stDepth << endl;
	cout << "stAlpha " << kernelOpt.stAlpha << endl;
	cout << "sFallof " << kernelOpt.sFallof << endl;
	cout << "albedoTreshold " << kernelOpt.albedoTreshold << endl;
	cout << "albedoDivide " << kernelOpt.albedoDivide << endl;
	cout << "sSpecularStrength " << kernelOpt.sSpecularStrength << endl;

	//kernel spatial final touch
	cout << "sfkernel " << kernelOpt.sfkernel << endl;
	cout << "sfradius " << kernelOpt.sfradius << endl;
	cout << "sfWeight " << kernelOpt.sfWeight << endl;
	cout << "sfColor " << kernelOpt.sfColor << endl;
	cout << "sfAlbedo " << kernelOpt.sfAlbedo << endl;
	cout << "sfNormal " << kernelOpt.sfNormal << endl;
	cout << "sfDepth " << kernelOpt.sfDepth << endl;
	cout << "sfFallof " << kernelOpt.sfFallof << endl;
	cout << "sfAlbedoDivide " << kernelOpt.sfAlbedoDivide << endl;
	cout << "sfSpecularStrength " << kernelOpt.sfSpecularStrength << endl;

	//kernel temporal
	cout << "tWeight " << kernelOpt.temporalWeight << endl;
	cout << "tkernel " << kernelOpt.tkernel << endl;
	cout << "tFallof " << kernelOpt.tFallof << endl;
	cout << "tBlockSize " << kernelOpt.tBlockSize << endl;
	cout << "tInterpolation " << kernelOpt.tInterpolation << endl;
	cout << "tSigmaColor " << kernelOpt.tSigmaColor << endl;
	cout << "tSigmaAlbedo " << kernelOpt.tSigmaAlbedo << endl;
	cout << "tMotionTreshold " << kernelOpt.tMotionTreshold << endl;
	cout << "tColorTreshold " << kernelOpt.tColorTreshold << endl;

	//kernel PWtemporal
	cout << "tPwWeight " << kernelOpt.temporalPwWeight << endl;
	cout << "tPwKernel " << kernelOpt.tPwKernel << endl;
	cout << "tPwFallof " << kernelOpt.tPwFallof << endl;
	cout << "tPwSearchRadius " << kernelOpt.tPwSearchRadius << endl;
	cout << "tPwSigmaColor " << kernelOpt.tPwSigmaColor << endl;
	cout << "tPwSigmaAlbedo " << kernelOpt.tPwSigmaAlbedo << endl;
	cout << "tPwSigmaDistance " << kernelOpt.tPwSigmaDistance << endl;
	cout << "tPwSpaceTreshold " << kernelOpt.tPwSpaceTreshold << endl;
	cout << "tPwIterations " << kernelOpt.tPwIterations << endl;

	//firefly filter
	cout << "ffkernel " << kernelOpt.ffkernel << endl;
	cout << "ffSigma " << kernelOpt.ffSigma << endl;
	cout << "ffGain " << kernelOpt.ffGain << endl;
	cout << "ffGamma " << kernelOpt.ffGamma << endl;
	cout << "ffRefractionStrange " << kernelOpt.ffRefractionStrange << endl;
	cout << "ffindirectSpecStrange " << kernelOpt.ffindirectSpecStrange << endl;
	cout << "" << endl;
}

int main(int argc, char* argv[])
{
	cout << "(c)D.Ginzburg denoiser, version 1.7" << endl;

	thread_data_t thr_data[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	initializeOptions(argc, argv, iOpt, kernelOpt);

	PrintFilterOpt();

	for (int frame = iOpt.startFrame; frame < iOpt.endFrame + 1; frame++)
		if (((iOpt.runMode == "single") && (frame == iOpt.runBlock)) || (iOpt.runMode == "multiple"))
		{
			for (int i = 0; i < iOpt.nCores; i++)
			{
				thr_data[i].frame = frame;
				thr_data[i].core = i;
			}
			cout << "Filtering frame " << FillZero(frame, 4) << endl;
			cout << "Read images.. ";
			readFrame(frame, pixels, iOpt);
			cout << "complete" << endl;

			int rc;
			clock_t start;
			double duration;
			start = std::clock();

			// unpremult
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, UnpremultImages, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Unpremult.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			cout << "complete" << endl;

			// NAN
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, NAN_check, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "NAN check.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			NAN_Copy(ResultBlockSmoothTMP, pixels);
			cout << "complete" << endl;

			// firefly
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, Firefly_Kill, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Firefly kill.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			cout << "complete" << endl;

			// motion compensation
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, MotionCompensation, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Motion compensation.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			cout << "complete" << endl;
			// block match MV
			cout << "Refine motion vectors.. ";
			BlockMatchMV();
			cout << "complete" << endl;
			// block match
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, BlockMatch, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Block match.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			cout << "complete" << endl;
			// filter01
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, NLM_First, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Main spatial filter.. ";

			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			TMP_Copy(ResultBlockSmoothTMP, ResultBlockSmooth);
			cout << "complete" << endl;
			// filter02
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, NLM_FinalTouch, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Final touch filter.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			TMP_Copy(ResultBlockSmoothTMP, ResultBlockSmooth);

			cout << "complete" << endl;
			// filter03
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, NLM_Temporal, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Temporal filter.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			TMP_Copy(ResultBlockSmoothTemporalTMP, ResultBlockSmoothTemporal);
			cout << "complete" << endl;

			// merge
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, Filter_merge, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Merge filters.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			cout << "complete" << endl;
			cout << "Write images.. ";
			WritePasses(frame);
			cout << "complete" << endl;
			// cleanup
			for (int i = 0; i < iOpt.nCores; i++)
			{
				rc = pthread_create(&threads[i], NULL, CleanupMemory, &thr_data[i]);

				if (rc)
				{
					cout << "Error:unable to create thread," << rc << endl;
					exit(-1);
				}
			}
			cout << "Cleanup memory.. ";
			for (int i = 0; i < iOpt.nCores; ++i)
				pthread_join(threads[i], NULL);
			cout << "complete" << endl;

			duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
			std::cout << "Complete. Denoise time: " << duration << " seconds" << '\n';
			cout << " " << endl;

		}

	return 0;
}