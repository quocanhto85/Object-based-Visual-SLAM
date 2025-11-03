#ifndef CUBOID_H
#define CUBOID_H

#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM3 {
struct Cuboid {
    std::string class_name;
    int class_id;
    float confidence;
    Eigen::Vector3f center; // 3D center position
    Eigen::Quaternionf rot; // Rotation quaternion
    Eigen::Vector3f dims; // Dimensions (width, height, length)
};
}

#endif