/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<sstream> // Required for stringstream

// 3D_CUBOID
#include <nlohmann/json.hpp>
#include "Cuboid.h"
// Eigen required for Vector3f and Quaternionf in Cuboid parsing
#include <Eigen/Core>
#include <Eigen/Geometry>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

// Forward declaration of utility function
void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5) // 3D_CUBOID
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence path_to_cuboids" << endl;
        return 1;
    }

    // Retrieve paths to images and timestamps
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);
    float imageScale = SLAM.GetImageScale();

    // 3D_CUBOID //
    // START: Cuboid Data Pre-Loading
    
    // The path to cuboid JSON files is passed as argv[4]
    string cuboids_dir = string(argv[4]);
    
    // --- INSERTED CODE START (replaces original Cuboid loading block) ---

    cout << "=== Loading Cuboids ===" << endl;
    cout << "Cuboids directory: " << cuboids_dir << endl;

    // Initialize vector to store ALL cuboids (one vector of cuboids per frame)
    vector<vector<ORB_SLAM3::Cuboid>> all_cuboids(nImages); 

    int frames_with_cuboids = 0;
    int total_cuboids_loaded = 0;

    try {
        for (int ni = 0; ni < nImages; ni++) {
            stringstream ss;
            // KITTI frames are 6-digit zero-padded (e.g., 000000.json)
            ss << setfill('0') << setw(6) << ni; 
            string json_path = cuboids_dir + "/" + ss.str() + ".json";
            
            ifstream ifs(json_path);
            if (!ifs.good()) {
                if (ni < 5) {  // Only show first few missing files
                    cout << "⚠ No JSON for frame " << ni << " at " << json_path << endl;
                }
                continue;
            }
            
            nlohmann::json j = nlohmann::json::parse(ifs);
            vector<ORB_SLAM3::Cuboid> frame_cubs;
            
            // Loop through each detected object in the JSON
            for (const auto& obj : j["objects"]) {
                ORB_SLAM3::Cuboid cub;
                cub.class_name = obj["class"];
                cub.class_id = obj["class_id"];
                cub.confidence = obj["confidence"];
                
                // Parse center (Eigen::Vector3f)
                auto cen = obj["center"];
                cub.center = Eigen::Vector3f(cen[0], cen[1], cen[2]);
                
                // Parse rotation (Eigen::Quaternionf: w, x, y, z)
                auto rot = obj["rotation"];
                cub.rot = Eigen::Quaternionf(rot["w"], rot["x"], rot["y"], rot["z"]);
                
                // Parse dimensions (Eigen::Vector3f)
                auto dim = obj["dimensions"];
                cub.dims = Eigen::Vector3f(dim[0], dim[1], dim[2]);
                
                // Filter by confidence threshold
                if (cub.confidence > 0.5) {
                    frame_cubs.push_back(cub);
                }
            }
            
            if (!frame_cubs.empty()) {
                all_cuboids[ni] = frame_cubs;
                frames_with_cuboids++;
                total_cuboids_loaded += frame_cubs.size();
                
                if (ni < 10) {  // Debug first 10 frames
                    cout << "✓ Frame " << ni << ": " << frame_cubs.size() << " cuboids" << endl;
                }
            }
        }
    } catch (const std::exception& e) {
        cerr << "Error loading JSON: " << e.what() << endl;
    }

    cout << "=== Cuboid Loading Complete ===" << endl;
    cout << "Frames with cuboids: " << frames_with_cuboids << "/" << nImages << endl;
    cout << "Total cuboids loaded: " << total_cuboids_loaded << endl;
    cout << "==============================" << endl;

    // Pass ALL cuboids to SLAM system BEFORE tracking loop
    cout << "Passing cuboids to SLAM system..." << endl;
    SLAM.SetCuboids(all_cuboids);


    // END: Cuboid Data Pre-Loading
    // ------------------------------------------------------------------------------------------------ //

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
#ifdef REGISTER_TIMES
    double t_resize = 0.f;
    double t_track = 0.f;
#endif

    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        // Declare t1 and t2 outside of ifdef blocks so they're always available
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

        // 3D_CUBOID //
        // START: Cuboid Data Per-Frame Update
        // ORIGINAL CODE REMOVED: Cuboids are now set once before the loop.
        // if (ni < all_cuboids.size()) {
        //     // Wrap single frame's cuboids to match expected signature
        //     vector<vector<ORB_SLAM3::Cuboid>> single_frame_wrapper;
        //     single_frame_wrapper.push_back(all_cuboids[ni]);
        //     SLAM.SetCuboids(single_frame_wrapper);
        // }
        // END: Cuboid Data Per-Frame Update
        // ------------------------------------------------------------------------------------------------ //

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe,vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenames[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}