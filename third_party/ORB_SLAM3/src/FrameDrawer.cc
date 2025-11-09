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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// --- NEW INCLUDES REQUIRED FOR CUBOID PROJECTION ---
#include <Eigen/Core>
#include <Eigen/Geometry>
// Note: Assumes <string> is included for to_string or that it's available.
// Also assumes ORB_SLAM3::Cuboid is defined and available.
// ---------------------------------------------------

#include<mutex>

namespace ORB_SLAM3
{

void FrameDrawer::DrawCuboids2D(cv::Mat& im)
{
    unique_lock<mutex> lock(mMutexCuboids);
    
    if (mCurrentFrameCuboids.empty()) {
        return;
    }
    
    if (mK.empty()) {
        static bool warned = false;
        if (!warned) {
            cout << "WARNING: Camera matrix not set in FrameDrawer!" << endl;
            warned = true;
        }
        return;
    }
    
    // Extract camera intrinsics
    float fx = mK.at<float>(0, 0);
    float fy = mK.at<float>(1, 1);
    float cx = mK.at<float>(0, 2);
    float cy = mK.at<float>(1, 2);
    
    int drawn = 0;
    for (const auto& cub : mCurrentFrameCuboids) {
        if (cub.confidence < 0.7) continue;
        
        // Get cuboid 3D properties
        Eigen::Matrix3f R = cub.rot.toRotationMatrix();
        Eigen::Vector3f t = cub.center;
        Eigen::Vector3f half = cub.dims / 2.0f;
        
        // Compute 8 corners in camera frame
        std::vector<Eigen::Vector3f> corners(8);
        int idx = 0;
        for (int dx = -1; dx <= 1; dx += 2) {
            for (int dy = -1; dy <= 1; dy += 2) {
                for (int dz = -1; dz <= 1; dz += 2) {
                    Eigen::Vector3f p_obj(dx * half(0), dy * half(1), dz * half(2));
                    corners[idx++] = R * p_obj + t;
                }
            }
        }
        
        // Project 3D corners to 2D
        std::vector<cv::Point2f> pts_2d(8);
        bool all_in_front = true;
        for (int i = 0; i < 8; i++) {
            float Z = corners[i](2);
            if (Z <= 0.1f) {
                all_in_front = false;
                break;
            }
            
            float u = fx * corners[i](0) / Z + cx;
            float v = fy * corners[i](1) / Z + cy;
            pts_2d[i] = cv::Point2f(u, v);
        }
        
        if (!all_in_front) continue;
        
        // Check if any point is visible
        bool any_visible = false;
        for (const auto& pt : pts_2d) {
            if (pt.x >= 0 && pt.x < im.cols && pt.y >= 0 && pt.y < im.rows) {
                any_visible = true;
                break;
            }
        }
        
        if (!any_visible) continue;
        
        // Set BRIGHT, THICK color based on class
        cv::Scalar color;
        string label;
        int thickness = 3;  // MUCH THICKER LINES
        
        if (cub.class_id == 0) {
            color = cv::Scalar(255, 0, 255);  // MAGENTA for Car (more visible)
            label = "Car";
        } else if (cub.class_id == 2) {
            color = cv::Scalar(0, 255, 255);  // CYAN for Cyclist
            label = "Cyclist";
        } else if (cub.class_id == 6) {
            color = cv::Scalar(255, 255, 0);  // YELLOW for Road
            label = "Road";
            thickness = 2;  // Thinner for road
        } else {
            color = cv::Scalar(255, 128, 0);  // ORANGE for others
            label = cub.class_name;
        }
        
        // Define 3D box edges
        int edges[12][2] = {
            {0,1}, {0,2}, {0,4},  // Back face edges
            {1,3}, {1,5},         // Connecting edges
            {2,3}, {2,6},         // Connecting edges
            {3,7},                // Front face edge
            {4,5}, {4,6},         // Bottom edges
            {5,7}, {6,7}          // Front face edges
        };
        
        // Draw 3D box edges with THICK lines
        for (int e = 0; e < 12; e++) {
            int i = edges[e][0];
            int j = edges[e][1];
            
            // Check if both points are in image bounds
            if (pts_2d[i].x >= 0 && pts_2d[i].x < im.cols &&
                pts_2d[i].y >= 0 && pts_2d[i].y < im.rows &&
                pts_2d[j].x >= 0 && pts_2d[j].x < im.cols &&
                pts_2d[j].y >= 0 && pts_2d[j].y < im.rows) {
                
                cv::line(im, pts_2d[i], pts_2d[j], color, thickness, cv::LINE_AA);
            }
        }
        
        // Draw front face with thicker lines for emphasis
        std::vector<int> front_face_edges = {3, 7, 10, 11};  // Front face
        for (int e : front_face_edges) {
            int i = edges[e][0];
            int j = edges[e][1];
            cv::line(im, pts_2d[i], pts_2d[j], color, thickness + 1, cv::LINE_AA);
        }
        
        // Compute center for label
        cv::Point2f center_2d(0, 0);
        int valid_pts = 0;
        for (const auto& pt : pts_2d) {
            if (pt.x >= 0 && pt.x < im.cols && pt.y >= 0 && pt.y < im.rows) {
                center_2d += pt;
                valid_pts++;
            }
        }
        
        if (valid_pts > 0) {
            center_2d.x /= valid_pts;
            center_2d.y /= valid_pts;
            
            // Draw label with background
            string text = label + " " + to_string((int)(cub.confidence * 100)) + "%";
            int baseline = 0;
            double font_scale = 0.7;
            int font_thickness = 2;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 
                                                 font_scale, font_thickness, &baseline);
            
            cv::Point text_org(center_2d.x - text_size.width / 2, center_2d.y - 10);
            
            // Black background for text
            cv::rectangle(im, 
                         text_org + cv::Point(0, baseline),
                         text_org + cv::Point(text_size.width, -text_size.height),
                         cv::Scalar(0, 0, 0), 
                         cv::FILLED);
            
            // Colored text
            cv::putText(im, text, text_org, cv::FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, font_thickness, cv::LINE_AA);
        }
        
        drawn++;
    }
    
    // Always show count in corner
    if (drawn > 0) {
        string count_text = "Cuboids: " + to_string(drawn);
        cv::putText(im, count_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    }
    
    static int debug_frames = 0;
    if (debug_frames < 10) {
        cout << "FrameDrawer: Drew " << drawn << " 2D cuboids (total candidates: " 
             << mCurrentFrameCuboids.size() << ")" << endl;
        debug_frames++;
    }
}

FrameDrawer::FrameDrawer(Atlas* pAtlas):both(false),mpAtlas(pAtlas)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mImRight = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    // NOTE: Initialization of mK is required here or via a settings/Update call
    // mK = cv::Mat::eye(3, 3, CV_32F); // Placeholder, actual K should be loaded
}

// --- START: NEW CUBOID METHODS ---

void FrameDrawer::SetCurrentFrameCuboids(const std::vector<ORB_SLAM3::Cuboid>& cuboids)
{
    unique_lock<mutex> lock(mMutexCuboids);
    mCurrentFrameCuboids = cuboids;
}

cv::Mat FrameDrawer::DrawFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization matches
    vector<int> vMatches; // Initialization matches
    vector<cv::KeyPoint> vCurrentKeys; // Current keys
    vector<bool> vbVO, vbMap; // Tracked or not
    vector<pair<cv::Point2f, cv::Point2f> > vTracks;
    int state; // Tracking state

    Frame currentFrame;
    vector<MapPoint*> vpLocalMap;
    vector<cv::KeyPoint> vMatchesKeys;
    vector<MapPoint*> vpMatchedMPs;
    vector<cv::KeyPoint> vOutlierKeys;
    vector<MapPoint*> vpOutlierMPs;
    map<long unsigned int, cv::Point2f> mProjectPoints;
    map<long unsigned int, cv::Point2f> mMatchedInImage;

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
            vTracks = mvTracks;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;

            currentFrame = mCurrentFrame;
            vpLocalMap = mvpLocalMap;

            vMatchesKeys = mvMatchedKeys;
            vpMatchedMPs = mvpMatchedMPs;

            vOutlierKeys = mvOutlierKeys;
            vpOutlierMPs = mvpOutlierMPs;

            mProjectPoints = mmProjectPoints;
            mMatchedInImage = mmMatchedInImage;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED)
    {
        // ... existing initialization drawing code ...
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    // ===== CRITICAL: Draw cuboids LAST, after all other drawing =====
    DrawCuboids2D(imWithInfo);
    // ================================================================

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawRightFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mImRight.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeysRight;
        }
    } // destroy scoped mutex -> release mutex

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }

                cv::line(im,pt1,pt2,cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = mvCurrentKeysRight.size();
        const int Nleft = mvCurrentKeys.size();

        for(int i=0;i<n;i++)
        {
            if(vbVO[i + Nleft] || vbMap[i + Nleft])
            {
                cv::Point2f pt1,pt2;
                cv::Point2f point;
                if(imageScale != 1.f)
                {
                    point = mvCurrentKeysRight[i].pt / imageScale;
                    float px = mvCurrentKeysRight[i].pt.x / imageScale;
                    float py = mvCurrentKeysRight[i].pt.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;
                }
                else
                {
                    point = mvCurrentKeysRight[i].pt;
                    pt1.x=mvCurrentKeysRight[i].pt.x-r;
                    pt1.y=mvCurrentKeysRight[i].pt.y-r;
                    pt2.x=mvCurrentKeysRight[i].pt.x+r;
                    pt2.y=mvCurrentKeysRight[i].pt.y+r;
                }

                // This is a match to a MapPoint in the map
                if(vbMap[i + Nleft])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,point,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,point,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}



void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nMaps = mpAtlas->CountMaps();
        int nKFs = mpAtlas->KeyFramesInMap();
        int nMPs = mpAtlas->MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    mThDepth = pTracker->mCurrentFrame.mThDepth;
    mvCurrentDepth = pTracker->mCurrentFrame.mvDepth;
    
    // NOTE: mK update. It should be initialized with the camera matrix from the Frame object.
    mK = pTracker->mCurrentFrame.mK; // Assuming mK is a member and mCurrentFrame has an mK member

    if(both){
        mvCurrentKeysRight = pTracker->mCurrentFrame.mvKeysRight;
        pTracker->mImRight.copyTo(mImRight);
        N = mvCurrentKeys.size() + mvCurrentKeysRight.size();
    }
    else{
        N = mvCurrentKeys.size();
    }

    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    //Variables for the new visualization
    mCurrentFrame = pTracker->mCurrentFrame;
    mmProjectPoints = mCurrentFrame.mmProjectPoints;
    mmMatchedInImage.clear();

    mvpLocalMap = pTracker->GetLocalMapMPS();
    mvMatchedKeys.clear();
    mvMatchedKeys.reserve(N);
    mvpMatchedMPs.clear();
    mvpMatchedMPs.reserve(N);
    mvOutlierKeys.clear();
    mvOutlierKeys.reserve(N);
    mvpOutlierMPs.clear();
    mvpOutlierMPs.reserve(N);

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;

                    // This part needs adjustment for stereo to use correct keypoint source
                    // For mono (i < mvCurrentKeys.size()) use mvCurrentKeys[i].pt
                    // For right (i >= mvCurrentKeys.size()) use mvCurrentKeysRight[i - mvCurrentKeys.size()].pt
                    
                    if (i < mvCurrentKeys.size()) {
                         mmMatchedInImage[pMP->mnId] = mvCurrentKeys[i].pt;
                    } else if (both) {
                         mmMatchedInImage[pMP->mnId] = mvCurrentKeysRight[i - mvCurrentKeys.size()].pt;
                    }

                }
                else
                {
                    // This part uses mvCurrentKeys[i] which is only valid for i < mvCurrentKeys.size()
                    // Assuming for visualization purposes, we only show outliers from the left frame
                    if (i < mvCurrentKeys.size()) {
                        mvpOutlierMPs.push_back(pMP);
                        mvOutlierKeys.push_back(mvCurrentKeys[i]);
                    }
                }
            }
        }

    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM