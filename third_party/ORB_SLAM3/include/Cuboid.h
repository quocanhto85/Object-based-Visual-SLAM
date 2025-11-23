/**
 * Cuboid.h
 *
 * 3D Cuboid structure for Object-based SLAM
 * Supports both regular object detection and ByteTrack tracking integration
 */

#ifndef CUBOID_H
#define CUBOID_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>

namespace ORB_SLAM3
{

    struct Cuboid
    {
        // Object identification
        std::string class_name; // e.g., "Car", "Person", "Cyclist"
        int class_id;           // Numeric class identifier
        float confidence;       // Detection confidence [0, 1]

        // Tracking information (ByteTrack integration)
        int track_id; // Unique tracking ID across frames (-1 if not tracked)

        // 3D geometry in camera coordinate frame
        Eigen::Vector3f center; // 3D center position (x, y, z) in meters
        Eigen::Quaternionf rot; // Rotation as quaternion
        Eigen::Vector3f dims;   // Dimensions (width, height, length) in meters

        // Constructor with default initialization
        Cuboid()
            : class_id(-1),
              confidence(0.0f),
              track_id(-1), // Default: no tracking
              center(Eigen::Vector3f::Zero()),
              rot(Eigen::Quaternionf::Identity()),
              dims(Eigen::Vector3f::Zero())
        {
        }

        // Constructor with parameters
        Cuboid(const std::string &name, int cls_id, float conf,
               const Eigen::Vector3f &cen, const Eigen::Quaternionf &rotation,
               const Eigen::Vector3f &dimensions, int tid = -1)
            : class_name(name),
              class_id(cls_id),
              confidence(conf),
              track_id(tid),
              center(cen),
              rot(rotation),
              dims(dimensions)
        {
        }

        // Check if cuboid has valid tracking information
        bool isTracked() const
        {
            return track_id > 0;
        }

        // Get formatted label string
        std::string getLabel() const
        {
            std::string label = class_name + " " +
                                std::to_string(static_cast<int>(confidence * 100)) + "%";
            if (isTracked())
            {
                label += " ID:" + std::to_string(track_id);
            }
            return label;
        }
    };

} // namespace ORB_SLAM3

#endif // CUBOID_H