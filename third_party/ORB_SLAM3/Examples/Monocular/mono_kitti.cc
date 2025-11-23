/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira,
 * Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream> // Required for stringstream

// 3D_CUBOID
#include <nlohmann/json.hpp>
#include "Cuboid.h"
// Eigen required for Vector3f and Quaternionf in Cuboid parsing
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;

// Forward declaration of utility function
void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if (argc != 5) // 3D_CUBOID
    {
        cerr << endl
             << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence path_to_cuboids" << endl;
        return 1;
    }

    // Retrieve paths to images and timestamps
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Enable cuboid export
    SLAM.EnableCuboidExport(true);

    // 3D_CUBOID //
    // START: Cuboid Data Pre-Loading with Enhanced Filtering

    string cuboids_dir = string(argv[4]);
    vector<vector<ORB_SLAM3::Cuboid>> all_cuboids(nImages);

    // Configuration for filtering
    const float CONFIDENCE_THRESHOLD = 0.7f; // Increased from 0.5 to reduce FPs
    const float MIN_DIMENSION = 0.5f;        // Minimum size (meters) to filter noise
    const float MAX_DIMENSION = 50.0f;       // Maximum size to filter outliers

    // Class-specific confidence thresholds
    map<string, float> class_thresholds = {
        {"car", 0.6f}, {"Car", 0.6f}, {"person", 0.7f}, {"Person", 0.7f}, {"pedestrian", 0.7f}, {"Pedestrian", 0.7f}, {"cyclist", 0.75f}, {"Cyclist", 0.75f}, {"bus", 0.8f}, {"Bus", 0.8f}, {"truck", 0.7f}, {"Truck", 0.7f}, {"traffic light", 0.8f}, {"Traffic Light", 0.8f}, {"traffic sign", 0.85f}, {"Traffic Sign", 0.85f}};

    // Classes to completely ignore (too unreliable)
    set<string> ignored_classes = {
        "Drivable Area", "drivable area",
        "Road", "road"};

    cout << "=== Loading Cuboids with Enhanced Filtering ===" << endl;
    cout << "Cuboids directory: " << cuboids_dir << endl;

    try
    {
        int total_loaded = 0;
        int total_filtered = 0;
        int frames_with_cuboids = 0;

        for (int ni = 0; ni < nImages; ni++)
        {
            stringstream ss;
            ss << setfill('0') << setw(6) << ni;
            string json_path = cuboids_dir + "/" + ss.str() + ".json";
            ifstream ifs(json_path);

            if (!ifs.good())
            {
                if (ni < 5)
                {
                    cout << "⚠ No JSON for frame " << ni << " at " << json_path << endl;
                }
                continue;
            }

            nlohmann::json j = nlohmann::json::parse(ifs);
            vector<ORB_SLAM3::Cuboid> frame_cubs;

            for (const auto &obj : j["objects"])
            {
                ORB_SLAM3::Cuboid cub;
                cub.class_name = obj["class"];
                cub.class_id = obj["class_id"];
                cub.confidence = obj["confidence"];
                
                // 3D_CUBOID: Parse track_id (ByteTrack integration)
                // Default to -1 if not present (backward compatibility with non-tracked JSON)
                if (obj.contains("track_id"))
                {
                    cub.track_id = obj["track_id"];
                }
                else
                {
                    cub.track_id = -1; // No tracking information available
                }

                // Skip ignored classes
                if (ignored_classes.count(cub.class_name) > 0)
                {
                    total_filtered++;
                    continue;
                }

                // Parse geometry
                auto cen = obj["center"];
                cub.center = Eigen::Vector3f(cen[0], cen[1], cen[2]);

                auto rot = obj["rotation"];
                cub.rot = Eigen::Quaternionf(rot["w"], rot["x"], rot["y"], rot["z"]);

                auto dim = obj["dimensions"];
                cub.dims = Eigen::Vector3f(dim[0], dim[1], dim[2]);

                // Apply class-specific confidence threshold
                float threshold = CONFIDENCE_THRESHOLD;
                if (class_thresholds.count(cub.class_name) > 0)
                    threshold = class_thresholds[cub.class_name];

                if (cub.confidence < threshold)
                {
                    total_filtered++;
                    continue;
                }

                // Filter by dimensions
                float max_dim = max({cub.dims[0], cub.dims[1], cub.dims[2]});
                float min_dim = min({cub.dims[0], cub.dims[1], cub.dims[2]});
                if (min_dim < MIN_DIMENSION || max_dim > MAX_DIMENSION)
                {
                    total_filtered++;
                    continue;
                }

                // Filter by distance
                float distance = cub.center.norm();
                if (distance > 100.0f)
                {
                    total_filtered++;
                    continue;
                }

                // Sanity check on quaternion
                if (abs(cub.rot.norm() - 1.0f) > 0.1f)
                {
                    total_filtered++;
                    continue;
                }

                frame_cubs.push_back(cub);
                total_loaded++;
            }

            if (!frame_cubs.empty())
            {
                all_cuboids[ni] = frame_cubs;
                frames_with_cuboids++;
            }

            // Debug output every 100 frames (with track_id statistics)
            if (ni % 100 == 0 && !frame_cubs.empty())
            {
                int tracked_count = 0;
                for (const auto& c : frame_cubs) {
                    if (c.track_id > 0) tracked_count++;
                }
                cout << "Frame " << ni << ": " << frame_cubs.size() << " cuboids kept, " 
                     << tracked_count << " with track_id" << endl;
            }
        }

        // Count total tracked cuboids
        int total_tracked = 0;
        for (const auto& frame_cubs : all_cuboids) {
            for (const auto& cub : frame_cubs) {
                if (cub.track_id > 0) total_tracked++;
            }
        }

        cout << "=== Cuboid Loading Complete ===" << endl;
        cout << "Frames with cuboids: " << frames_with_cuboids << "/" << nImages << endl;
        cout << "Total cuboids loaded: " << total_loaded << endl;
        cout << "Total with track_id: " << total_tracked << " (" 
             << (total_loaded > 0 ? (100.0f * total_tracked / total_loaded) : 0) << "%)" << endl;
        cout << "Total filtered out: " << total_filtered << endl;
        cout << "=================================" << endl;
    }
    catch (const std::exception &e)
    {
        cerr << "Error loading JSON: " << e.what() << endl;
    }

    // Pass ALL cuboids to SLAM system BEFORE tracking loop
    cout << "Passing cuboids to SLAM system..." << endl;
    SLAM.SetCuboids(all_cuboids);

    // END: Enhanced Cuboid Data Pre-Loading
    // ------------------------------------------------------------------------------------------------ //

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

#ifdef REGISTER_TIMES
    double t_resize = 0.f;
    double t_track = 0.f;
#endif

    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        // im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        cv::Mat im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_COLOR);
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
            t_resize = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe, vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenames[ni]);

        // Save cuboids for this frame
        SLAM.SaveCurrentCuboids(ni, "../../../../data/predicted_cuboids/");

        // --- Scale correction block (every 100 frames) ---
        // if (ni % 100 == 0) // Adjust interval as needed
        // {
        //     // If you have GPS or ground truth scale
        //     float true_scale = GetGroundTruthScale(ni);
        //     SLAM.CorrectScale(true_scale);
        // }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    SLAM.Shutdown();
    // SLAM.SmoothTrajectory();   // Apply moving average filter
    // SLAM.OptimizeTrajectory(); // Global optimization

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
        totaltime += vTimesTrack[ni];

    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    // SLAM.SaveKeyFrameTrajectoryKITTI("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
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

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}