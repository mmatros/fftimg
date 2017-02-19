#include <iostream>
#include <CImg.h>
#include <fftw3.h>
#include <math.h>
#include <complex>
#include <cassert>

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

void calculatePhaseAndMagnitude(double* phase, double* magnitude, fftw_complex* data, int width, int height){
        if (phase == nullptr || magnitude == nullptr) {
                return;
        }
        for (size_t i  = 0; i < width * height; ++i) {
                double re = (double)data[i][0]/(double)(width * height);
                double im = (double)data[i][1]/(double)(width * height);
                std::complex<double> c(re,im);
                magnitude[i] = (double)sqrt(re*re + im*im);
                phase[i] = std::arg(c) + M_PI;
                phase[i] = (phase[i] / (double)(2 * M_PI)) * 255;
        }
        swapQuadrants(width, phase);
        swapQuadrants(width, magnitude);
}

int main(int argc, char** argv){
        // CImg<double> image("../data/lena_grey.bmp");
        CImg<double> image("../data/sea.bmp");
        CImg<double> fragIn("../data/sea_frag.bmp");
        int width = image.width();
        int height = image.height();
        int frag_width = fragIn.width();
        int frag_height = fragIn.height();
        std::cout<<"width: "<<width<<" height: "<<height<<std::endl;
        assert(width==height);

        double* bytes = image.data();
        double* bytes_frag = fragIn.data();

        fftw_complex *big, *big_fft, *frag, *frag_fft, *corr;
        fftw_plan planBig, planFrag, planCorr;
        big = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );
        frag = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );
        big_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );
        frag_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );
        corr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height );

        double median = fragIn.median();

        for (int i = 0; i <width; i++ ) {
                for (int j = 0; j < height; j++) {
                        big[i*width+j][0] = bytes[i*width+j];
                        if (i < frag_width && j < frag_height) {
                                double val = fragIn[i*frag_width+j] - median;
                                if (val > 0) {
                                        frag[i*width+j][0] = val;
                                }else{
                                        frag[i*width+j][0] = 0;
                                }
                        }else{
                                frag[i*width+j][0] = 0;
                        }
                }
        }

        planBig = fftw_plan_dft_2d(width, height, big, big_fft, FFTW_FORWARD, FFTW_ESTIMATE );
        planFrag = fftw_plan_dft_2d(width, height, frag, frag_fft, FFTW_FORWARD, FFTW_ESTIMATE );
        planCorr = fftw_plan_dft_2d(width, height, frag_fft, corr, FFTW_BACKWARD, FFTW_ESTIMATE );

        fftw_execute(planBig);
        fftw_execute(planFrag);

        for (size_t i  = 0; i < width * height; ++i) {
                std::complex<double> val1(frag_fft[i][0], frag_fft[i][1]* -1);
                std::complex<double> val2(big_fft[i][0], big_fft[i][1]);
                std::complex<double> val3 = val2 * val1;
                frag_fft[i][0] = val3.real();
                frag_fft[i][1] = val3.imag();
        }

        double *outBytes = new double[width * height];
        fftw_execute(planCorr);
        for (size_t i  = 0; i < width * height; ++i) {
                outBytes[i] = (double)corr[i][0]/(double)(width * height);
        }

        CImg<double> imageCorr(outBytes, width, height);
        imageCorr = imageCorr.normalize(0,255);
        imageCorr.display("Correlation");

        delete outBytes;
        fftw_destroy_plan(planBig);
        fftw_destroy_plan(planFrag);
        fftw_destroy_plan(planCorr);
        fftw_free(big);
        fftw_free(big_fft);
        fftw_free(frag);
        fftw_free(frag_fft);
        fftw_free(corr);

        fftw_cleanup();

        return 0;
}
