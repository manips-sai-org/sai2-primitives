/**
 * @file ForceSensor.cpp
 * @author William Chong (wmchong@stnaford.edu)
 * @brief 
 * @version 0.1
 * @date 2024-02-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ForceSensor.h"

namespace Sai2Primitives {

ForceMeasurement ForceSensor::getCalibratedForceMoment(const Vector3d& raw_force, const Vector3d& raw_moment) {
    Vector3d compensated_force = raw_force;
    Vector3d compensated_moment = raw_moment;
    
    // bias compensation 
    compensated_force -= _force_bias;
    compensated_moment -= _moment_bias;

    // gravity compensation 
    Matrix3d R_world_to_link;  // rotate vector from link frame to world frame 
    R_world_to_link = _robot->rotation(_link_name);
    Matrix3d R_world_to_sensor;  // rotate vector from sensor frame to world frame 
    R_world_to_sensor = R_world_to_link * _T_link_to_sensor.linear();  // rotate vector from sensor to world frame 
    Vector3d rotated_gravity = R_world_to_sensor.transpose() * _gravity;
    Vector3d p_tool_sensor_frame = _mass * rotated_gravity;
    compensated_force += p_tool_sensor_frame;
    compensated_moment += _com.cross(p_tool_sensor_frame);

    // inertial compensation
    Vector6d vel = _robot->velocity6dInWorld(_link_name, _T_link_to_sensor.translation());
    _robot->updateKinematicsCustom(false, false, true, true);
    Vector6d accel = _robot->acceleration6dInWorld(_link_name, _T_link_to_sensor.translation());
    _robot->updateModel();
    Vector3d linear_vel_local_frame = R_world_to_sensor.transpose() * vel.head(3);
    Vector3d angular_vel_sensor_frame = R_world_to_sensor.transpose() * vel.tail(3);
    Vector3d linear_accel_sensor_frame = R_world_to_sensor.transpose() * accel.head(3);
    Vector3d angular_accel_sensor_frame = R_world_to_sensor.transpose() * accel.tail(3);
    compensated_force -= _mass * linear_accel_sensor_frame - \
                            angular_accel_sensor_frame.cross(_mass * _com) - \
                            angular_vel_sensor_frame.cross(angular_vel_sensor_frame.cross(_mass * _com));
    compensated_moment -= _inertia * angular_accel_sensor_frame - \
                            angular_vel_sensor_frame.cross(_inertia * angular_vel_sensor_frame) - \
                            (_mass * _com).cross(linear_accel_sensor_frame);
                            
    return ForceMeasurement{compensated_force, compensated_moment};
}

}  // namespace 