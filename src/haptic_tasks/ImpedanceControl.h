/*
 * ImpedanceControl.h
 *
 *    This controller implements the classic impedance bilateral scheme for haptic teleoperation.
 * 		The desired force feedback is computed for the haptic device (force feedback) and the desired position and rotation
 *    are evaluated for the robot (desired_position_robot & desired_rotation_robot).
 * 		The impedance-type controller can additionally include plane/line/orientation guidances and virtual workspace limits.
 *
 *     Author: Margot Vulliez
 */

 #ifndef SAI2_IMPEDANCE_CONTROL_H_
 #define SAI2_IMPEDANCE_CONTROL_H_

 #include "Sai2Model.h"
 #include "filters/ButterworthFilter.h"
 #include <Eigen/Dense>

 using namespace std;
 using namespace Eigen;

 namespace Sai2Primitives
 {

 class ImpedanceControl
 {
// friend class HapticPassivityController; ?

 public:

   ////////////////////////////////////////////////////////////////////////////////////////////////////
   //// Constructor, Destructor and Initialization of the impedance-type controller
   ////////////////////////////////////////////////////////////////////////////////////////////////////
   	/**
   	 * @brief Constructor  This constructor creates the haptic impedance controller for a simple bilateral teleoperation scheme.
   	 * @details:
     *     The center of the robot task workspace is given in robot frame
     *		 Device specifications are used to adjust controller gains and to saturate commanded force/torque to limit values
     *     The rotation matrix from haptic device operational space to robot operational space is defined
     *
   	 * @param center_position_robot 			    The task home position of the robot in its operational space
   	 * @param center_rotation_robot     		  The task home orientation of the robot in its operational space
     * @param max_device_force      Maximal force, torque, and gripper force the device can render
     * @param max_device_stiffness 	Maximal translational and rotational stiffness the device can render
     * @param max_device_damping 		Maximal translational and rotational damping the device can render
   	 * @param Transform_Matrix_DeviceToRobot 	Rotation matrix from haptic device to robot frame
   	 *
   	 */
   	ImpedanceControl(const Vector3d center_position_robot,
     		            const Matrix3d center_rotation_robot,
                    const Vector3d max_device_force,
                    const Vector2d max_device_stiffness,
                    const Vector2d max_device_damping,
     		            const Matrix3d Rotation_Matrix_DeviceToRobot = Matrix3d::Identity());

   	/**
   	 * @brief Detructor  This destructor deletes the pointers and stop the haptic controller.
   	 */
   	~ImpedanceControl();

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Impedance-type bilateral controller - Core Methods (Data updates and commands' computation)
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Computes the commands for the haptic device and the robot from the impedance-type controller.
     * @details:
     *			computeCommands - The haptic commands are computed from the robot sensed task force.
     *                            The desired robot position/rotation are evaluated from the haptic device current position.
     *                            All commands are given in proper robot/haptic operational spaces.
     *      computeGripperCommands - The gripper haptic feedback is computed from the robot sensed gripper force.
     *                               Desired robot gripper position is evaluated.
 	   *
 	   *		 !! Make sure to update the haptic device and robot data (position, sensed force) before calling this function !!
     *        Update methods must be used to provide the controller with needed sensory data.
 	   *
 	   * @param desired_position_robot    		  The desired position of the controlled robot
 	   * @param desired_rotation_robot    		  The desired orientation of the controlled robot
     * @param commanded_force_device          The desired force feedback of the haptic device
 	   * @param commanded_torque_device         The desired torque feedback of the haptic device
     * @param desired_gripper_position_robot  The desired position of the robot gripper
     * @param commanded_gripper_force_device  The desired force applied to the device gripper
 	   */
    void computeCommands(Vector3d& desired_position_robot,
  								         Matrix3d& desired_rotation_robot,
                           Vector3d& commanded_force_device,
                           Vector3d& commanded_torque_device);
    void computeGripperCommands(double& desired_gripper_position_robot, double& commanded_gripper_force_device);

     /**
      * @brief Send desired haptic feedback to the device
      * @details:
      *			If set to true, the task forces are rendered through the haptic feedback. Otherwise, the device is only compensating for its own gravity.
      *
      * @param feedback_OnOff      Send haptic feedback : true or false
      */
      void sendFeedback(bool feedback_OnOff = true);

     /**
      * @brief Update robot sensed task force, torque and gripper force, in robot frame.
  	   * @details:
  	   *		 The robot interaction force sensory data are updated to compute the haptic feedback.
       *     Force sensor data are filtered if specified thanks to the setFilterOn() additional method.
  	   *
  	   * @param sensed_task_force     		   Robot sensed task force
  	   * @param sensed_task_torque    		   The desired orientation of the controlled robot
       * @param sensed_task_gripper_force    The desired force feedback of the haptic device
  	   */
     	void updateRobotSensedForce(const Vector3d sensed_task_force = Vector3d::Zero(), const Vector3d sensed_task_torque = Vector3d::Zero(),
                            const double sensed_task_gripper_force = 0.0);

      /**
       * @brief Update Haptic device position data (position, rotation, and gripper position), in haptic frame.
        * @details:
        *		 The haptic device sensed position/Rotation and gripper position are updated to compute the robot commands.
        *
        * @param current_position_device     		   Haptic device current operational space position
        * @param current_rotation_device    		   Haptic device current operational space rotation matrix
        * @param current_position_gripper_device   Haptic device current gripper position
        */
       void updateDevicePosition(const Vector3d current_position_device = Vector3d::Zero(), const Matrix3d current_rotation_device = Matrix3d::Identity(),
                                 const double current_position_gripper_device = 0.0);

       /**
        * @brief Update Haptic device velocity data (translational and rotational device velocities and gripper velocity), in haptic frame.
         * @details:
         *		 The haptic device sensed end-effector and gripper velocities are updated to compute the haptic guidance damping term.
         *     Velocity update is only needed to compute haptic guidance methods and while using gripper as switch
         *
         * @param current_trans_velocity_device     Haptic device current end-effector translational velocity
         * @param current_rot_velocity_device       Haptic device current end-effector rotational velocity
         * @param current_gripper_velocity_device   Haptic device current gripper velocity
         */
        void updateDeviceVelocity(const Vector3d current_trans_velocity_device = Vector3d::Zero(), const Vector3d current_rot_velocity_device = Vector3d::Zero(),
                                  const double current_gripper_velocity_device = 0.0);


     ////////////////////////////////////////////////////////////////////////////////////////////////////
     // Impedance-type bilateral controller - Specification Methods (To change controller factors, or device/robot task origin)
     ////////////////////////////////////////////////////////////////////////////////////////////////////
     /**
   	 * @brief Set the center of the device workspace in device frame
   	 *
   	 * @param center_position_device     Center position of the haptic device in its operational space
   	 * @param center_rotation_device     Center orientation of the haptic device's end-effector in its operational space
   	 */
   	void setDeviceCenter(const Vector3d center_position_device = Vector3d::Zero(),
   		            	    const Matrix3d center_rotation_device = Matrix3d::Identity());

   	/**
   	 * @brief Set the center of the robot task workspace in robot frame
   	 *
   	 * @param center_position_robot     The robot task center position in its operational space
   	 * @param center_rotation_robot     The robot task center orientation in its operational space
   	 */
   	void setRobotCenter(const Vector3d center_position_robot,
   		            	    const Matrix3d center_rotation_robot);

   	/**
   	 * @brief Set the rotation matrix from haptic device operational space to robot operational space
   	 *
   	 * @param Rotation_Matrix_DeviceToRobot    Rotation matrix from device to robot world frames
   	 */
   	void setDeviceRobotRotation(const Matrix3d Rotation_Matrix_DeviceToRobot = Matrix3d::Identity());

    /**
     * @brief Set haptic device specifications (max stiffness, max damping, and max force/torque/gripper force)
     * @details:
     *		 Device specifications are used to adjust controller gains and to saturate commanded force/torque to limit values
     *
     * @param max_device_force      Maximal force, torque, and gripper force the device can render
     * @param max_device_stiffness 	Maximal translational and rotational stiffness the device can render
     * @param max_device_damping 		Maximal translational and rotational damping the device can render
     */
     void setDeviceMaxSpecification (const Vector3d max_device_force = Vector3d::Zero(),
                                     const Vector2d max_device_stiffness = Vector2d::Zero(),
                                     const Vector2d max_device_damping = Vector2d::Zero());

    /**
     * @brief Set the scaling factors from the device workspace to the robot task workspace
     * @details:
     *		 Device sensed motion are multiplied by the scaling factor to get the desired robot motion.
     *     Robot sensed force are divided by the scaling factor to compute the haptic feedback.
     *
     * @param scaling_factor_trans      Translational scaling factor
     * @param scaling_factor_rot       	Rotational scaling factor
     * @param scaling_factor_gripper    Gripper position scaling factor
     */
    void setScalingFactors(const double scaling_factor_trans = 1.0, const double scaling_factor_rot = 1.0, const double scaling_factor_gripper = 1.0);

    /**
  	 * @brief Set an additional reduction factors between the sensed task force and the haptic rendered force
     * @details:
     *		 This reduction factor is applied to robot scaled force to improve user perception or comfort.
     *
  	 * @param reduction_factor_torque_feedback 		Feedback force reduction factor
  	 * @param reduction_factor_force_feedback 		Feedback torque reduction factor
     * @param reuction_factor_gripper_force       Feedback gripper force reduction factor
  	 */
  	void setReductionFactorForceFeedback (const double reduction_factor_force_feedback = 1.0, const double reduction_factor_torque_feedback = 1.0,
                                          const double reduction_factor_gripper_force = 1.0);

    /**
  	 * @brief Setting methods to add a virtual line or plane as haptic guidance (in haptic frame)
     * @details:
     *    setHapticGuidanceGains - sets the stiffness and damping parameters for the haptic guidance
     *    setGuidancePlane - define a 3D plane using a point and a normal vector to apply guidance
     *    setGuidanceLine - define a line in space through two points to apply guidance
  	 *
  	 * @param guidance_stiffness 	Stiffness of the virtual guidance
  	 * @param guidance_damping 		Damping of the virtual guidance
     * @param planeGuidance_OnOff Set true to apply plane guidance
  	 * @param plane_origin_point coordinate vector of the origin point
  	 * @param plane_normal_vec normal vector for the plane
     * @param lineGuidance_OnOff Set true to apply line guidance
  	 * @param line_first_point vector to first point from world origin
  	 * @param line_second_point vector to second point from world origin
  	 */
    void setHapticGuidanceGains(const double guidance_stiffness, const double guidance_damping);
    void setGuidancePlane(bool planeGuidance_OnOff, const Vector3d plane_origin_point = Vector3d::Zero(), const Vector3d plane_normal_vec = Vector3d(0.0, 0.0, 1.0));
    void setGuidanceLine(bool lineGuidance_OnOff, const Vector3d line_first_point = Vector3d::Zero(), const Vector3d line_second_point = Vector3d(0.0, 0.0, 1.0));

     ////////////////////////////////////////////////////////////////////////////////////////////////////
     // Impedance-type bilateral controller - Additional Methods
     ////////////////////////////////////////////////////////////////////////////////////////////////////
     /**
   	 * @brief Set a filter on robot force sensor data at the given normalized cut-off frequencies (between 0 and 0.5)
   	 *
     * @param filter_OnOff                       Filter sensed force : true or false
   	 * @param cutOff_frequency_force            Cut-off frequency of the sensed force (default value : 0.05)
   	 * @param cutOff_frequency_moment           Cut-off frequency of the sensed torque (default value : 0.05)
     * @param cutOff_frequency_gripper_force    Cut-off frequency of the sensed gripper force (default value : 0.05)
   	 */
     void setFilterOn(bool filter_OnOff = false, const double cutOff_frequency_force = 0.06,
                      const double cutOff_frequency_moment = 0.06, const double cutOff_frequency_gripper_force = 0.06);

      /**
    	 * @brief Add virtual limits to render the device workspace boundary through the force feedback
    	 * @details:
       *    The size of the device workspace is defined by its equivalent sphere radius in [m] and the maximum tilt angle in [rad].
    	 *
       * @param virtualWS_OnOff                  Add virtual workspace limits : true or false
    	 * @param device_workspace_radius_limit   Radius of the smallest sphere including the haptic device Workspace
    	 * @param device_workspace_angle_limit   	Maximum tilt angle of the haptic device
    	 */
    	void setWorkspaceLimits(bool virtualWS_OnOff = false, double device_workspace_radius_limit = 0.1, double device_workspace_angle_limit = 90*M_PI/180.0);

      /**
    	 * @brief A method to use the gripper like a switch, must be called cyclically
    	 * @details:
       *    Return gripper state (boolean value) with respect to gripper position, and send constant force to push back the gripper
    	 *
       * @param gripper_state         Value of gripper switch : true or false
       * @param commanded_gripper_force_device  The desired force applied to the device gripper
    	 */
      void useGripperAsSwitch(bool& gripper_state, double& commanded_gripper_force_device);

private:

     /**
      * @brief Compute guidance force for set virtual fixtures
      * @details:
      *    Guidance plane or line can be added to the haptic controllers thanks to setPlane() and setLine().
     	*    Computes guidance force to feedback to the user in haptic frame
     	*/
     	void ComputePlaneGuidanceForce();
     	void ComputeLineGuidanceForce();


      ///////////////////////////////////////////////////////////////////////////////////
      // Controller parameters an data, specifiec through setting methods
      ///////////////////////////////////////////////////////////////////////////////////
      //// Controller status and operational modes ////
    	bool _send_haptic_feedback; // If set to false, send 0 forces and torques to the haptic device
      bool _filter_on; // Filter force sensor data
      bool _add_workspace_virtual_limit; // add a virtual sphere delimiting the haptic device workspace

    	bool _enable_plane_guidance; // add guidance along a user-defined plane
    	bool _enable_line_guidance; // add guidance along a user-defined plane

      bool gripper_init; // Gripper initialization status

      //// Haptic device variables ////
    	// Haptic device specifications
    	double _max_linear_stiffness_device, _max_angular_stiffness_device;
    	double _max_linear_damping_device, _max_angular_damping_device;
    	double _max_force_device, _max_torque_device, _max_gripper_force_device;

    	double _device_workspace_radius_limit;
    	double _device_workspace_angle_limit;

      // Desired haptic feedback
    	Vector3d _commanded_force_device;
    	Vector3d _commanded_torque_device;
    	double _commanded_gripper_force_device;

      // Workspace scaling factors in translation and rotation (applied to force and motion)
      double _scaling_factor_trans, _scaling_factor_rot, _scaling_factor_gripper;
      // Additional force feedback factors
      double _reduction_factor_force_feedback, _reduction_factor_torque_feedback, _reduction_factor_gripper_force;

    	// Haptic device position
    	Vector3d _current_position_device;
    	Matrix3d _current_rotation_device;
    	double _current_position_gripper_device;

      // Haptic device velocity
      Vector3d _current_trans_velocity_device;
      Vector3d _current_rot_velocity_device;
      double _current_gripper_velocity_device;

      // Haptic device center position and orientation
      Vector3d _home_position_device;
      Matrix3d _home_rotation_device;

    	//// Robot variables ////
    	//Desired robot position
    	Vector3d _desired_position_robot;
    	Matrix3d _desired_rotation_robot;
      double _desired_gripper_position_robot;

      // Workspace center for the task in robot frame
      Vector3d _center_position_robot;
      Matrix3d _center_rotation_robot;

      //Transformation matrix from the device frame to the robot frame
      Matrix3d _Rotation_Matrix_DeviceToRobot;

    	// Sensed task force
    	Vector3d _sensed_task_force;
      Vector3d _sensed_task_torque;
      double _sensed_task_gripper_force;

      // Sensed force filters
      Sai2Common::ButterworthFilter* _force_filter;
      Sai2Common::ButterworthFilter* _moment_filter;
      Sai2Common::ButterworthFilter* _gripper_force_filter;
      double _cutOff_frequency_force;
      double _cutOff_frequency_moment;
      double _cutOff_frequency_gripper_force;

      //// Haptic guidance variables ////
    	// Haptic guidance gains
    	double _guidance_stiffness;
    	double _guidance_damping;
    	// Guidance plane parameters
    	Vector3d _plane_origin_point;
    	Vector3d _plane_normal_vec;
    	Vector3d _guidance_force_plane;
    	// Guidance line parameters
    	Vector3d _guidance_line_vec;
    	Vector3d _line_first_point;
    	Vector3d _line_second_point;
    	Vector3d _guidance_force_line;

};
}  /* namespace Sai2Primitives */

/* SAI2_IMPEDANCE_CONTROL_H_ */
#endif
