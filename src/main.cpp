#include <iostream>
#include <CImg.h>
#include <fftw3.h>
#include <math.h>
#include <complex>

using namespace cimg_library;

void swapQuadrants(int squareSize, double* image){
	int half = floor(squareSize / 2.0);

	for(int i = 0; i < half; i++) {
		for(int j = 0; j < half; j++) {
			int upper = j + (squareSize * i);
			int lower = upper + (squareSize * half) + half;

      double tmp = image[upper];
      image[upper] = image[lower];
      image[lower] = tmp;

      tmp = image[upper + half];
      image[upper+half] = image[lower - half];
      image[lower - half] = tmp;
		}
	}
}

int main(int argc, char** argv){
  CImg<double> image("../data/lena_grey.bmp");
  int width = image.width();
  int height = image.height();
  std::cout<<"width: "<<width<<" height: "<<height<<std::endl;

  double* bytes = image.data();

  fftw_complex *in, *out, *result;
  fftw_plan planA, planB;
  in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );
  for (size_t i = 0; i< width * height; ++i){
    in[i][0] = bytes[i];
  }

  out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );
  result = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );
  planA = fftw_plan_dft_2d(width, height, in, out, FFTW_FORWARD, FFTW_ESTIMATE );
  planB = fftw_plan_dft_2d(width, height, out, result, FFTW_BACKWARD, FFTW_ESTIMATE );

  fftw_execute(planA);
  fftw_execute(planB);

  double *outBytes = new double[width * height];
  double *magnitude = new double[width * height];
  double *phase = new double[width * height];
  for (size_t i  = 0 ; i < width * height; ++i){
    outBytes[i] = (double)result[i][0]/(double)(width * height);
    double re = (double)out[i][0]/(double)(width * height);
    double im = (double)out[i][1]/(double)(width * height);
    std::complex<double> c(re,im);
    magnitude[i] = (double)sqrt(re*re + im*im);
    phase[i] = std::arg(c) + M_PI;
    phase[i] = (phase[i] / (double)(2 * M_PI)) * 255;
  }

  swapQuadrants(width, magnitude);
  swapQuadrants(width, phase);

  CImg<double> imageOut(outBytes, width, height);
  imageOut.display("Converted");

  CImg<double> imageMagnitude(magnitude, width, height);
  imageMagnitude.display("Magnitude");
  imageMagnitude.save("../data/magnitude.bmp");

  CImg<double> imagePhase(phase,  width, height);
  imagePhase.display("Phase");
  imageMagnitude.save("../data/phase.bmp");


  delete outBytes;
  delete magnitude;
  delete phase;
  fftw_destroy_plan(planA);
  fftw_destroy_plan(planB);
  fftw_free(in);
  fftw_free(out);
  fftw_free(result);

  return 0;
}
