// Shared-library entry around Ericsson ETCPACK v2.74 (not included here; build_etcpack.sh fetches it).
// int etcpack_rgba8(rgba, W, H, out, fast, perceptual): raw RGBA8 in, 16 bytes per 4x4 block out (8 EAC alpha + 8 ETC2 colour).
#include <sys/timeb.h>
#define _timeb timeb
#define _ftime ftime
#define main etcpack_main
#include "ETCPACK/source/etcpack.cxx"
#undef main
static int etcpack_inited = 0;
extern "C" int etcpack_rgba8(const unsigned char *rgba, int W, int H, unsigned char *out, int fast, int perceptual)
{
	if (W % 4 || H % 4) return 2;
	if (!etcpack_inited) { readCompressParams(); setupAlphaTableAndValtab(); etcpack_inited = 1; }
	codec = CODEC_ETC2; format = ETC2PACKAGE_RGBA_NO_MIPMAPS; verbose = 0; formatSigned = 0;
	speed = fast ? SPEED_FAST : SPEED_SLOW; metric = perceptual ? METRIC_PERCEPTUAL : METRIC_NONPERCEPTUAL;
	size_t n = (size_t)W * H;
	uint8 *img = (uint8 *)malloc(n * 3), *imgdec = (uint8 *)malloc(n * 3), *alphaimg = (uint8 *)malloc(n);
	if (!img || !imgdec || !alphaimg) return 1;
	for (size_t i = 0; i < n; i++) { img[3 * i] = rgba[4 * i]; img[3 * i + 1] = rgba[4 * i + 1]; img[3 * i + 2] = rgba[4 * i + 2]; alphaimg[i] = rgba[4 * i + 3]; }
	for (int y = 0; y < H / 4; y++)
		for (int x = 0; x < W / 4; x++)
		{
			unsigned int b1, b2;
			if (metric == METRIC_NONPERCEPTUAL)
			{
				if (speed == SPEED_FAST) compressBlockETC2Fast(img, alphaimg, imgdec, W, H, 4 * x, 4 * y, b1, b2);
				else compressBlockETC2Exhaustive(img, imgdec, W, H, 4 * x, 4 * y, b1, b2);
			}
			else
			{
				if (speed == SPEED_FAST) compressBlockETC2FastPerceptual(img, imgdec, W, H, 4 * x, 4 * y, b1, b2);
				else compressBlockETC2ExhaustivePerceptual(img, imgdec, W, H, 4 * x, 4 * y, b1, b2);
			}
			if (speed == SPEED_SLOW) compressBlockAlphaSlow(alphaimg, 4 * x, 4 * y, W, H, out);
			else compressBlockAlphaFast(alphaimg, 4 * x, 4 * y, W, H, out);
			out[8] = b1 >> 24; out[9] = b1 >> 16; out[10] = b1 >> 8; out[11] = b1; out[12] = b2 >> 24; out[13] = b2 >> 16; out[14] = b2 >> 8; out[15] = b2;
			out += 16;
		}
	free(img); free(imgdec); free(alphaimg);
	return 0;
}
