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


#include "Viewer.h"
#include <pangolin/pangolin.h>
#include <nlohmann/json.hpp> // 3D_CUBOID
#include <mutex>

namespace ORB_SLAM3
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath, Settings* settings):
    both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    mbStopTrack = false;
}

void Viewer::newParameterLoader(Settings *settings) {
    mImageViewerScale = 1.f;

    float fps = settings->fps();
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::Size imSize = settings->newImSize();
    mImageHeight = imSize.height;
    mImageWidth = imSize.width;

    mImageViewerScale = settings->imageViewerScale();
    mViewpointX = settings->viewPointX();
    mViewpointY = settings->viewPointY();
    mViewpointZ = settings->viewPointZ();
    mViewpointF = settings->viewPointF();
}

bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    mImageViewerScale = 1.f;

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::FileNode node = fSettings["Camera.width"];
    if(!node.empty())
    {
        mImageWidth = node.real();
    }
    else
    {
        std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if(!node.empty())
    {
        mImageHeight = node.real();
    }
    else
    {
        std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.imageViewScale"];
    if(!node.empty())
    {
        mImageViewerScale = node.real();
    }

    node = fSettings["Viewer.ViewpointX"];
    if(!node.empty())
    {
        mViewpointX = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointX parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointY"];
    if(!node.empty())
    {
        mViewpointY = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointY parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointZ"];
    if(!node.empty())
    {
        mViewpointZ = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointZ parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointF"];
    if(!node.empty())
    {
        mViewpointF = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointF parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    // cv::namedWindow("ORB-SLAM3: Current Frame", cv::WINDOW_FREERATIO);

    pangolin::CreateWindowAndBind("ORB-SLAM3: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuCamView("menu.Camera View",false,false);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",false,true);
    pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial Graph",true,true);
    pangolin::Var<bool> menuShowOptLba("menu.Show Local BA",false,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
    pangolin::Var<bool> menuStop("menu.Stop",false,false);
    pangolin::Var<bool> menuStepByStep("menu.Step By Step",false,true);
    pangolin::Var<bool> menuStep("menu.Step",false,false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc, Ow;
    Twc.SetIdentity();
    Ow.SetIdentity();
    
    cv::Mat currentCameraPose = cv::Mat::eye(4,4,CV_32F);

    bool bFollow = true;
    bool bLocalizationMode = false;
    bool bStepByStep = false;
    bool bCameraView = true;

    if(mpTracker->mSensor == mpSystem->MONOCULAR || mpTracker->mSensor == mpSystem->IMU_MONOCULAR)
    {
        menuShowGraph = true;
    }

    float trackedImageScale = mpTracker->GetImageScale();

    // cout << "trackedImageScale: " << trackedImageScale << endl;

    bool both = false;
    if(mpTracker->mSensor == mpSystem->System::STEREO || mpTracker->mSensor == mpSystem->System::IMU_STEREO ||
       mpTracker->mSensor == mpSystem->System::IMU_RGBD || mpTracker->mSensor == mpSystem->System::RGBD )
        both = true;

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc,Ow);
        
        // ========== ADD THIS BLOCK: Store camera pose for DrawCurrentCuboids ==========
        // Extract the 4x4 transformation matrix from Twc
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                currentCameraPose.at<float>(i,j) = Twc.m[j*4 + i];  // Note: OpenGL uses column-major
            }
        }
        
        // Store in member variable with thread safety
        {
            unique_lock<mutex> lock(mMutexCamera);
            mCameraPose = currentCameraPose.clone();
        }
        // ===============================================================================

        if(mbStopTrack)
        {
            menuStepByStep = true;
            mbStopTrack = false;
        }

        if(menuFollowCamera && bFollow)
        {
            if(bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if(menuFollowCamera && !bFollow)
        {
            if(bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10, 0,0,0,0.0,0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
        }

        if(menuTopView && mpMapDrawer->mpAtlas->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,50, 0,0,0,0.0,0.0, 1.0));
            s_cam.Follow(Ow);
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }

        if(menuStepByStep && !bStepByStep)
        {
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        }
        else if(!menuStepByStep && bStepByStep)
        {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if(menuStep)
        {
            mpTracker->mbStep = true;
            menuStep = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph || menuShowInertialGraph || menuShowOptLba)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph, menuShowInertialGraph, menuShowOptLba);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        // ========== ADD THIS: Draw 3D cuboids ==========
        // DrawCurrentCuboids();
        // ===============================================

        pangolin::FinishFrame();

        cv::Mat toShow;
        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);

        if(both){
            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);
            cv::hconcat(im,imRight,toShow);
        }
        else{
            toShow = im;
        }

        if(mImageViewerScale != 1.f)
        {
            int width = toShow.cols * mImageViewerScale;
            int height = toShow.rows * mImageViewerScale;
            cv::resize(toShow, toShow, cv::Size(width, height));
        }

        cv::imshow("ORB-SLAM3: Current Frame",toShow);
        cv::waitKey(mT);

        if(menuReset)
        {
            menuShowGraph = true;
            menuShowInertialGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->ResetActiveMap();
            menuReset = false;
        }

        if(menuStop)
        {
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();

            // Stop all threads
            mpSystem->Shutdown();

            // Save camera trajectory
            mpSystem->SaveTrajectoryEuRoC("CameraTrajectory.txt");
            mpSystem->SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
            menuStop = false;
        }

        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

// 3D_CUBOID
void Viewer::SetCuboids(const std::vector<std::vector<ORB_SLAM3::Cuboid>>& cuboids) {
    unique_lock<mutex> lock(mMutexCuboids);
    mCuboids = cuboids;
    cout << "Viewer::SetCuboids - Received " << cuboids.size() << " frames of cuboid data" << endl;
    if(mpMapDrawer) {
        mpMapDrawer->SetCuboids(cuboids);
    }
}
// 3D_CUBOID
void Viewer::SetCurrentFrameId(unsigned long id) {
    std::unique_lock<std::mutex> lock(mMutexFrameId);
    mCurrentFrameId = id;
}

// 3D_CUBOID
// void Viewer::DrawCurrentCuboids() {
//     unsigned long currentId = mpMapDrawer->GetCurrentFrameId();
//     if (mCuboids.empty() || currentId >= mCuboids.size()) return;

//     const auto& cubs = mCuboids[currentId];

//     cv::Mat Tcw = mCameraPose.clone();
//     cv::Mat Twc = cv::Mat::eye(4, 4, CV_32F);
//     Twc.rowRange(0, 3).colRange(0, 3) = Tcw.rowRange(0, 3).colRange(0, 3).t();
//     Twc.rowRange(0, 3).col(3) = -Twc.rowRange(0, 3).colRange(0, 3) * Tcw.rowRange(0, 3).col(3);
//     Twc.at<float>(3, 0) = Twc.at<float>(3, 1) = Twc.at<float>(3, 2) = 0;
//     Twc.at<float>(3, 3) = 1;

//     Eigen::Matrix4f eTwc;
//     for (int i = 0; i < 4; ++i)
//         for (int j = 0; j < 4; ++j)
//             eTwc(i, j) = Twc.at<float>(i, j);

//     for (const auto& cub : cubs) {
//         if (cub.confidence < 0.5) continue;  // Filter low-confidence

//         float r = 0.0f, g = 0.0f, b = 0.0f;
//         if (cub.class_id == 0) { r = 0.0; g = 0.0; b = 1.0; }  // Car: blue
//         else if (cub.class_id == 2) { r = 1.0; g = 0.0; b = 0.0; }  // Cyclist: red
//         else if (cub.class_id == 6) { r = 0.5; g = 0.5; b = 0.5; }  // Drivable Area: gray
//         else continue;
//         glColor3f(r, g, b);

//         Eigen::Matrix3f R = cub.rot.toRotationMatrix();
//         Eigen::Vector3f t = cub.center;
//         Eigen::Vector3f half = cub.dims / 2.0f;

//         std::vector<Eigen::Vector3f> corners_cam(8);
//         int idx = 0;
//         for (int dx = -1; dx <= 1; dx += 2)
//             for (int dy = -1; dy <= 1; dy += 2)
//                 for (int dz = -1; dz <= 1; dz += 2) {
//                     Eigen::Vector3f p_obj(dx * half(0), dy * half(1), dz * half(2));
//                     corners_cam[idx++] = R * p_obj + t;
//                 }

//         std::vector<Eigen::Vector3f> corners_world(8);
//         for (int i = 0; i < 8; ++i) {
//             Eigen::Vector4f p_cam(corners_cam[i](0), corners_cam[i](1), corners_cam[i](2), 1.0f);
//             Eigen::Vector4f p_world = eTwc * p_cam;
//             corners_world[i] = p_world.head<3>();
//         }

//         int edges[12][2] = {{0,1},{0,2},{0,4},{1,3},{1,5},{2,3},{2,6},{3,7},{4,5},{4,6},{5,7},{6,7}};
//         glLineWidth(2.0f);
//         glBegin(GL_LINES);
//         for (int e = 0; e < 12; ++e) {
//             int i = edges[e][0], j = edges[e][1];
//             glVertex3f(corners_world[i](0), corners_world[i](1), corners_world[i](2));
//             glVertex3f(corners_world[j](0), corners_world[j](1), corners_world[j](2));
//         }
//         glEnd();
//     }
// }

void Viewer::DrawCurrentCuboids() 
{
    // Get current frame ID
    unsigned long currentId;
    {
        unique_lock<mutex> lock(mMutexFrameId);
        currentId = mCurrentFrameId;
    }
    
    // Get cuboids for current frame
    vector<ORB_SLAM3::Cuboid> cubs;
    {
        unique_lock<mutex> lock(mMutexCuboids);
        
        if (mCuboids.empty()) {
            static bool warned = false;
            if (!warned) {
                cout << "WARNING: mCuboids is empty in DrawCurrentCuboids!" << endl;
                warned = true;
            }
            return;
        }
        
        if (currentId >= mCuboids.size()) {
            return;
        }
        
        cubs = mCuboids[currentId];
        
        if (cubs.empty()) {
            return;
        }
        
        // Debug output for first few frames
        static int debug_count = 0;
        if (debug_count < 5) {
            cout << "DrawCurrentCuboids: Frame " << currentId << " has " << cubs.size() << " cuboids" << endl;
            debug_count++;
        }
    }

    // Get camera pose
    cv::Mat Tcw;
    {
        unique_lock<mutex> lock(mMutexCamera);
        if (mCameraPose.empty()) {
            cout << "WARNING: Camera pose is empty!" << endl;
            return;
        }
        Tcw = mCameraPose.clone();
    }
    
    if (Tcw.rows != 4 || Tcw.cols != 4) {
        cout << "ERROR: Invalid camera pose dimensions!" << endl;
        return;
    }

    // Compute Twc (world to camera)
    cv::Mat Twc = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Twc.rowRange(0,3).colRange(0,3) = Rwc;
    Twc.rowRange(0,3).col(3) = -Rwc * tcw;

    // Convert to Eigen
    Eigen::Matrix4f eTwc;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            eTwc(i, j) = Twc.at<float>(i, j);

    // Draw each cuboid
    int drawn = 0;
    for (const auto& cub : cubs) {
        if (cub.confidence < 0.5) continue;

        // Set color based on class
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (cub.class_id == 0) { 
            r = 0.0; g = 0.0; b = 1.0;  // Car: blue
        } else if (cub.class_id == 2) { 
            r = 1.0; g = 0.0; b = 0.0;  // Cyclist: red
        } else if (cub.class_id == 5) { 
            r = 1.0; g = 0.75; b = 0.8;  // Traffic Light: pink
        } else if (cub.class_id == 8) { 
            r = 0.0; g = 1.0; b = 0.0;  // Bus: green
        } else if (cub.class_id == 6) { 
            r = 0.5; g = 0.5; b = 0.5;  // Drivable: gray
        } else {
            continue;
        }
        
        glColor3f(r, g, b);

        // Get cuboid parameters
        Eigen::Matrix3f R = cub.rot.toRotationMatrix();
        Eigen::Vector3f t = cub.center;
        Eigen::Vector3f half = cub.dims / 2.0f;

        // Compute 8 corners in camera frame
        std::vector<Eigen::Vector3f> corners_cam(8);
        int idx = 0;
        for (int dx = -1; dx <= 1; dx += 2) {
            for (int dy = -1; dy <= 1; dy += 2) {
                for (int dz = -1; dz <= 1; dz += 2) {
                    Eigen::Vector3f p_obj(dx * half(0), dy * half(1), dz * half(2));
                    corners_cam[idx++] = R * p_obj + t;
                }
            }
        }

        // Transform to world frame
        std::vector<Eigen::Vector3f> corners_world(8);
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector4f p_cam(corners_cam[i](0), corners_cam[i](1), corners_cam[i](2), 1.0f);
            Eigen::Vector4f p_world = eTwc * p_cam;
            corners_world[i] = p_world.head<3>();
        }

        // Draw cuboid edges
        int edges[12][2] = {
            {0,1},{0,2},{0,4},{1,3},{1,5},{2,3},
            {2,6},{3,7},{4,5},{4,6},{5,7},{6,7}
        };
        
        glLineWidth(3.0f);
        glBegin(GL_LINES);
        for (int e = 0; e < 12; ++e) {
            int i = edges[e][0], j = edges[e][1];
            glVertex3f(corners_world[i](0), corners_world[i](1), corners_world[i](2));
            glVertex3f(corners_world[j](0), corners_world[j](1), corners_world[j](2));
        }
        glEnd();
        
        drawn++;
    }
    
    static int frame_draw_count = 0;
    if (frame_draw_count < 5 && drawn > 0) {
        cout << "Drew " << drawn << " cuboids for frame " << currentId << endl;
        frame_draw_count++;
    }
}


/*void Viewer::SetTrackingPause()
{
    mbStopTrack = true;
}*/

}
