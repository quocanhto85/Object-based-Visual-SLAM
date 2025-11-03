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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

// --- START: NEW INCLUDES FOR CUBOIDS ---
#ifdef __APPLE__
    // macOS - but this won't be used in Docker container
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
    // Linux/Unix - this is what you need in Docker
    #include <GL/gl.h>
    #include <GL/glu.h>
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Cuboid.h"
// Note: Assumes "Cuboid.h" is included in "MapDrawer.h" or available via another common header.
// --- END: NEW INCLUDES FOR CUBOIDS ---

namespace ORB_SLAM3
{


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
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
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));

    }

    glEnd();
    
    // --- INSERTED CODE: Draw Cuboids ---
    // DrawCuboids();  // COMMENTED OUT - causes chaotic lines
    // -----------------------------------
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}

// 3D_CUBOID
void MapDrawer::SetCurrentFrameId(unsigned long id) {
    std::unique_lock<std::mutex> lock(mMutexFrameId);
    mCurrentFrameId = id;
}

// 3D_CUBOID
unsigned long MapDrawer::GetCurrentFrameId() const {
    std::unique_lock<std::mutex> lock(mMutexFrameId);
    return mCurrentFrameId;
}

// --- START: NEW CUBOID METHODS ---

void MapDrawer::SetCuboids(const std::vector<std::vector<ORB_SLAM3::Cuboid>>& cuboids) {
    unique_lock<mutex> lock(mMutexCuboids);
    mCuboids = cuboids;
}

void MapDrawer::DrawCuboids() {
    unique_lock<mutex> lock(mMutexCuboids);
    
    const unsigned long currentFrameIdx = GetCurrentFrameId(); 

    if(mCuboids.empty())
        return;
    
    // Check if current frame index is valid
    if(currentFrameIdx >= mCuboids.size())
        return;
    
    glLineWidth(3.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // MODIFIED: Only draw cuboids from the CURRENT frame (not all frames)
    const vector<ORB_SLAM3::Cuboid>& frameCuboids = mCuboids[currentFrameIdx];
    
    // Set full opacity for current frame cuboids
    float alpha = 1.0f;
    
    for(const auto& cuboid : frameCuboids) {
            // Skip low confidence detections
            if(cuboid.confidence < 0.3)
                continue;
                
            // Set color based on class with alpha
            if(cuboid.class_name == "Car" || cuboid.class_name == "car") {
                glColor4f(1.0f, 0.0f, 0.0f, alpha); // Red
            } 
            else if(cuboid.class_name == "Cyclist" || cuboid.class_name == "cyclist") {
                glColor4f(0.0f, 1.0f, 0.0f, alpha); // Green
            }
            else if(cuboid.class_name == "Pedestrian" || cuboid.class_name == "pedestrian" || 
                    cuboid.class_name == "Person" || cuboid.class_name == "person") {
                glColor4f(0.0f, 0.0f, 1.0f, alpha); // Blue
            }
            else if(cuboid.class_name == "Van" || cuboid.class_name == "van") {
                glColor4f(1.0f, 0.5f, 0.0f, alpha); // Orange
            }
            else if(cuboid.class_name == "Truck" || cuboid.class_name == "truck") {
                glColor4f(0.5f, 0.0f, 0.5f, alpha); // Purple
            }
            else if(cuboid.class_name == "Traffic Light" || cuboid.class_name == "traffic light") {
                glColor4f(1.0f, 1.0f, 0.0f, alpha); // Yellow
            }
            else if(cuboid.class_name == "Traffic Sign" || cuboid.class_name == "traffic sign") {
                glColor4f(0.0f, 1.0f, 1.0f, alpha); // Cyan
            }
            else {
                glColor4f(0.7f, 0.7f, 0.7f, alpha); // Gray for others
            }
            
            // Get cuboid parameters
            Eigen::Vector3f center = cuboid.center;
            Eigen::Vector3f dims = cuboid.dims;
            Eigen::Quaternionf rotation = cuboid.rot;
            
            // Create 8 corners of the cuboid in local coordinates
            Eigen::Vector3f corners[8];
            corners[0] = Eigen::Vector3f(-dims[0]/2, -dims[1]/2, -dims[2]/2);
            corners[1] = Eigen::Vector3f( dims[0]/2, -dims[1]/2, -dims[2]/2);
            corners[2] = Eigen::Vector3f( dims[0]/2,  dims[1]/2, -dims[2]/2);
            corners[3] = Eigen::Vector3f(-dims[0]/2,  dims[1]/2, -dims[2]/2);
            corners[4] = Eigen::Vector3f(-dims[0]/2, -dims[1]/2,  dims[2]/2);
            corners[5] = Eigen::Vector3f( dims[0]/2, -dims[1]/2,  dims[2]/2);
            corners[6] = Eigen::Vector3f( dims[0]/2,  dims[1]/2,  dims[2]/2);
            corners[7] = Eigen::Vector3f(-dims[0]/2,  dims[1]/2,  dims[2]/2);
            
            // Transform corners to world coordinates
            for(int i = 0; i < 8; i++) {
                corners[i] = rotation * corners[i] + center;
            }
            
            // Draw cuboid wireframe
            glBegin(GL_LINES);
            
            // Bottom face
            glVertex3f(corners[0][0], corners[0][1], corners[0][2]);
            glVertex3f(corners[1][0], corners[1][1], corners[1][2]);
            
            glVertex3f(corners[1][0], corners[1][1], corners[1][2]);
            glVertex3f(corners[2][0], corners[2][1], corners[2][2]);
            
            glVertex3f(corners[2][0], corners[2][1], corners[2][2]);
            glVertex3f(corners[3][0], corners[3][1], corners[3][2]);
            
            glVertex3f(corners[3][0], corners[3][1], corners[3][2]);
            glVertex3f(corners[0][0], corners[0][1], corners[0][2]);
            
            // Top face
            glVertex3f(corners[4][0], corners[4][1], corners[4][2]);
            glVertex3f(corners[5][0], corners[5][1], corners[5][2]);
            
            glVertex3f(corners[5][0], corners[5][1], corners[5][2]);
            glVertex3f(corners[6][0], corners[6][1], corners[6][2]);
            
            glVertex3f(corners[6][0], corners[6][1], corners[6][2]);
            glVertex3f(corners[7][0], corners[7][1], corners[7][2]);
            
            glVertex3f(corners[7][0], corners[7][1], corners[7][2]);
            glVertex3f(corners[4][0], corners[4][1], corners[4][2]);
            
            // Vertical edges
            glVertex3f(corners[0][0], corners[0][1], corners[0][2]);
            glVertex3f(corners[4][0], corners[4][1], corners[4][2]);
            
            glVertex3f(corners[1][0], corners[1][1], corners[1][2]);
            glVertex3f(corners[5][0], corners[5][1], corners[5][2]);
            
            glVertex3f(corners[2][0], corners[2][1], corners[2][2]);
            glVertex3f(corners[6][0], corners[6][1], corners[6][2]);
            
            glVertex3f(corners[3][0], corners[3][1], corners[3][2]);
            glVertex3f(corners[7][0], corners[7][1], corners[7][2]);
            
            glEnd();
            
            // Draw diagonal lines on front face for better visibility (assuming front is +Z)
            glLineWidth(1.0f);
            glBegin(GL_LINES);
            glVertex3f(corners[4][0], corners[4][1], corners[4][2]);
            glVertex3f(corners[6][0], corners[6][1], corners[6][2]);
            glVertex3f(corners[5][0], corners[5][1], corners[5][2]);
            glVertex3f(corners[7][0], corners[7][1], corners[7][2]);
            glEnd();
            glLineWidth(3.0f);
        }
    
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}

// --- END: NEW CUBOID METHODS ---

} //namespace ORB_SLAM