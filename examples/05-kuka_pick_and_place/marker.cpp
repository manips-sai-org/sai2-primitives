#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;

int main (int argc, char** argv) {
    // create marker
    Mat markerImg;
    Ptr<aruco::Dictionary> dictionary =
       aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(aruco::DICT_6X6_250));
    int markerId = 11;
    int markerSize = 200;
    int borderBits = 1;
    string out_path("resources/marker.png");
    aruco::drawMarker(dictionary, markerId, markerSize, markerImg, borderBits);
    imshow("marker", markerImg);
    waitKey(0);
    imwrite(out_path.c_str(), markerImg);
    
    // // match marker corners
    // string image_path("resources/env0.jpg");
    // Mat image = imread(image_path.c_str(), IMREAD_COLOR);
    // Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    // vector< vector< Point2f > > corners;
    // vector< int > ids;
    // Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    // aruco::detectMarkers(image, dictionary, corners, ids, params);
    
    // // draw matched markers
    // aruco::drawDetectedMarkers(image, corners, ids);

    // // print matched corners
    // for (int i = 0; i < corners.size(); i++) {
    //     cout << "corner[i]: ";
    //     for (int j = 0; j < corners.size(); j++) {
    //         cout << corners[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // // estimate marker poses
    // Mat camera_matrix = (Mat1d(3, 3) << 1., 0., 0.5, 0., 1., 0.5, 0., 0., 1.);
    // Mat distortion_coeff = (Mat1d(1, 4) << 0., 0., 0., 0.);
    // std::vector<cv::Vec3d> rvecs, tvecs;
    // aruco::estimatePoseSingleMarkers(corners, 0.1, camera_matrix, distortion_coeff, rvecs, tvecs);
    
    // // draw axis
    // for(int i = 0; i < ids.size(); i++)
    //     cv::aruco::drawAxis(image, camera_matrix, distortion_coeff, rvecs[i], tvecs[i], 0.1);
    
    // // display image
    // namedWindow("Display window", WINDOW_NORMAL);
    // cvResizeWindow("Display window", 500, 500);
    // imshow("Display window", image);
    // waitKey(0);
    // return 0;
}