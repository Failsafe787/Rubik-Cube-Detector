// The "Rubik's Cube Detector" program.
// It loads several images sequentially and tries to find rubik's cubes in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
using namespace cv;
using namespace std;

class MapArea {
    public:
    vector<vector<Point> > areas;
    vector<int> area_size;
    vector<vector<Point> > return_areas;
    vector<vector<Point> > rubikcube(){
        for(int i=0; i<area_size.size(); i++){
            int counter = 1;
            for(int j=i+1; j<area_size.size(); j++){
                if(area_size.at(i)==area_size.at(j)){
                    counter++;
                }
            }
            if(counter > 10){
                for(int k=0; k<area_size.size(); k++)
                    if(area_size.at(k)==area_size.at(i))
                        return_areas.push_back(areas.at(k));
                cout << "È molto probabile che ci sia un cubo di Rubik nell'immagine!" << endl;
                return return_areas;
            }
        }
        cout << "Non sembra sia presente un cubo di Rubik nell'immagine. Prova a cambiare prospettiva!" << endl;
        return return_areas;
    }
    void addsquare(vector<Point> square){
        areas.push_back(square);
        area_size.push_back(getSquareArea(square));
    }
    void clear(){
        areas.clear();
        area_size.clear();
        return_areas.clear();
    }
    private:
    int getSquareArea(vector<Point> quadrangle){
        double distance1 = sqrt(pow((quadrangle.at((1)).x - quadrangle.at(0).x), 2) + pow((quadrangle.at((1)%4).y - quadrangle.at(0).y), 2));
        double distance2 = sqrt(pow((quadrangle.at((2)%4).x - quadrangle.at(1).x), 2) + pow((quadrangle.at((2)%4).y - quadrangle.at(1).y), 2));
        int size_area = distance1*distance2;
        int approxarea = size_area + 1000/2;
        approxarea -= approxarea % 1000;
        return approxarea;
    }
};

void help()
{
	cout <<
	"\nA program that allows the detection of a Rubik's cube inside a picture\n"
	"Call:\n"
	"./squares \"image1_path\" \"image2_path\" ... \"imageN_path\"\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}


int thresh = 25, N = 11;
const char* wndname = "Square Detection Demo";
Mat image;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

bool iscube(vector<Point> quadrangle){
    bool valid = true;
    for(int i=0; i<3; i++){
        if(valid){
        double distance1 = sqrt(pow((quadrangle.at((i+1)).x - quadrangle.at(i).x), 2) + pow((quadrangle.at((i+1)%4).y - quadrangle.at(i).y), 2));
        double distance2 = sqrt(pow((quadrangle.at((i+2)%4).x - quadrangle.at(i+1).x), 2) + pow((quadrangle.at((i+2)%4).y - quadrangle.at(i+1).y), 2));
        if((min(distance1,distance2)/max(distance1,distance2))<0.6)
            valid = false;
        }
    }
    return valid;
}



Mat correctGamma( Mat img, double gamma ) {
 double inverse_gamma = 1.0 / gamma;

 Mat lut_matrix(1, 256, CV_8U );
 uchar * ptr = lut_matrix.ptr();
 for( int i = 0; i < 256; i++ )
   ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

 Mat result;
 LUT( img, lut_matrix, result );

 return result;
}

void lowpassGaussianFilter(double D, Mat& Filter)
{
//    order: oder of filter
//    D: diamter of filter cutoff frequency 0.607 of maximum value
//    Filter: matrix of the filter

     double dist;
     const int times = 100;

     for (int i=0; i<Filter.rows; i++)
     {
         for (int j=0; j<Filter.cols; j++)
         {
           dist = sqrt( (i-Filter.rows/2)*(i-Filter.rows/2) + (j-Filter.cols/2)*(j-Filter.cols/2) );

           Filter.at<float>(i,j) = exp(-dist*dist/(2*D*D));

         }
     }
}

//-------------------------------------------------------------------
// logTransformation()
//-------------------------------------------------------------------
//This function determines the log() transformation applied
// on the Spectrum determined to enhance its visibility.


void logTransformation(Mat& input)
{
    input += Scalar::all(1);
    log(input, input);
    normalize(input, input, 0, 1, CV_MINMAX);
}

void shiftSpectrum(Mat& input)
{

    // crop the spectrum, if it has an odd number of rows or columns
    //complexImg = complexImg(Rect(0, 0, complexImg.cols & -2, complexImg.rows & -2));

    int cx = input.cols/2;
    int cy = input.rows/2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center
    Mat tmp;
    Mat q0(input, Rect(0, 0, cx, cy));
    Mat q1(input, Rect(cx, 0, cx, cy));
    Mat q2(input, Rect(0, cy, cx, cy));
    Mat q3(input, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

}

Mat lowpassFilter(Mat &image){
    Mat gtimg, padded, complexImg, complexFilter;
    cvtColor(image, gtimg, CV_RGB2GRAY);
    int M = getOptimalDFTSize( gtimg.rows );
    int N = getOptimalDFTSize( gtimg.cols );
    copyMakeBorder(gtimg, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planes, 2 , complexImg);
    dft(complexImg, complexImg); //discrete Fourier Trasform
    shiftSpectrum(complexImg); //shift the Spectrum in the image centre
    Mat Filter(complexImg.size(), CV_32F);
    lowpassGaussianFilter(100,Filter);
    // precomputing of the Spectrum
    Mat planesFilter[] = {Filter, Filter};
    merge(planesFilter, 2, complexFilter);
    // multiply the filter * signal spectrum
    multiply(complexImg, complexFilter, complexImg); //Calculates the per-element scaled product of two arrays.
    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat mag = planes[0];
    logTransformation(mag);
    //now we reverse the process:
    //reallocate the spectrum not centered!!
    shiftSpectrum(complexImg); //shift the Spectrum in the image centre
    dft(complexImg, complexImg, DFT_INVERSE || DFT_SCALE);
    split(complexImg, planes);
    //magnitude(planes[0], planes[1], planes[0]); //usually it also possible skip
                                                  //the determination of magnitude()
                                                  //because the j part is negligible
    //visualize the rebuilded image
    Mat outMagt = planes[0];
    Mat outMag = outMagt(Rect(0,0,image.cols, image.rows));
    normalize(outMag, outMag, 0, 1, CV_MINMAX);
    return outMag;
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void findSquares( const Mat& image, MapArea &rubik_candidates )
{
    Mat timg, gray0(image.size(), CV_8U), gray;

    //Saturate the image to obtain more defined edges
    //Mat cimg = Mat::zeros( image.size(), image.type() );
    //for( int y = 0; y < image.rows; y++ ){
    //    for( int x = 0; x < image.cols; x++ ){
    //        for( int c = 0; c < 3; c++ ){
    //            cimg.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( 2.5*( image.at<Vec3b>(y,x)[c] ) + 70 );
    //         }
    //    }
    //}

    //Gamma-correction
    //Mat cimg = correctGamma(image, 3.0);

    // down-scale and upscale the image to filter out the noise, really faster than Gaussian ;-)
    GaussianBlur( image, timg, Size( 3, 3 ), 0, 0 );
    Mat outMagt = lowpassFilter(timg);
    vector<vector<Point> > contours;
    cvtColor(outMagt, outMagt, CV_GRAY2RGB);
    timg.convertTo(outMagt, CV_8U);
    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)

                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential - morphological operator which fills the holes
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list, I obtain 4 points
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ ) //Cerco triangoli e rettangoli
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.95 && iscube(approx))
                        rubik_candidates.addsquare(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA); //NOTA: lo spessore della linea è elevato, non ho solo una, ma più linee per riconoscere forma
    }
}

void rotatePerspective( Mat& image, vector<Point2f> points ){
    vector<Point2f> tpoints;
    Mat outImage;
    int width  = 200;
    int height = 200;
    tpoints.push_back(Point2f(0,0));
    tpoints.push_back(Point2f(tpoints.at(0).x + width, tpoints.at(0).y));
    tpoints.push_back(Point2f(tpoints.at(0).x, tpoints.at(0).y + height));
    tpoints.push_back(Point2f(tpoints.at(0).x + width, tpoints.at(0).y + height));
    Mat_<double> homography = findHomography(points, tpoints, 0);
    warpPerspective(image, outImage, homography, outImage.size());
    MapArea rubik_candidates;
    findSquares(outImage, rubik_candidates);
    drawSquares(outImage, rubik_candidates.rubikcube());
    imshow("Rotated", outImage);
}

void onMouse(int evt, int x, int y, int flags, void* param) {
    if(evt == CV_EVENT_LBUTTONDOWN) {
        vector<Point2f>* ptPtr = (vector<Point2f>*)param;
        ptPtr -> push_back(Point(x,y));

    if (ptPtr -> size() > 3) //we have 4 points
        {
            for (int i=0; i<ptPtr -> size(); i++){
                cout << "X and Y coordinates are given below" << endl;
                cout << (ptPtr -> at(i)).x << " - " << (ptPtr -> at(i)).y <<endl;
            }
            rotatePerspective(image, *ptPtr);
            ptPtr -> clear();
        }
    }
}


int main(int argc, char** argv)
{
    if(argc<2){
        help();
    }
    else{
        namedWindow( wndname, CV_WINDOW_AUTOSIZE);
        vector<vector<Point> > squares;
        vector<Point2f> points;
        setMouseCallback(wndname, onMouse, (void*)&points);
        MapArea rubik_candidates;
        for( int i = 1; i<argc ; i++ )
        {
            points.clear();
            image = imread(argv[i], 1);
            if( image.empty() )
            {
                cout << "Couldn't load " << argv[i] << endl;
                continue;
            }
            rubik_candidates.clear();
            findSquares(image, rubik_candidates);
            Mat toScreen = image.clone();
            drawSquares(toScreen, rubik_candidates.rubikcube());
            imshow(wndname, toScreen);
            int c = waitKey();
            if( (char)c == 27 )
                break;
            if( (char)c == 114)
                break;
        }
    }

    return 0;
}
