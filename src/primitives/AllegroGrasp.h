/*
 * AllegroGrasp.h
 *
 *      This class creates a motion primitive for a redundant arm using a posori task and a joint task in its nullspace
 *
 *      Author: Mikael Jorda
 */

#ifndef SAI2_PRIMITIVES_ALLEGRO_GRASP_H_
#define SAI2_PRIMITIVES_ALLEGRO_GRASP_H_

#include "redis/RedisClient.h"
#include "tasks/PosOriTask.h"
#include "tasks/JointTask.h"
#include "primitives/RedundantArmMotion.h"

namespace Sai2Primitives
{

class AllegroGrasp
{
public:

	/**
	 * @brief Constructor that takes the control frame in local link frame as an affine transform matrix
	 * 
	 * @param robot          robot model
	 * @param link_name      link to which are attached the control frame and the sensor frame (end effector link)
	 * @param control_frame  Transformation matrix of the control frame description in link frame
	 */
	AllegroGrasp(RedisClient* redis_client,
					   Sai2Model::Sai2Model* robot,
					   const std::string link_name,
	                   const Eigen::Affine3d control_frame = Eigen::Affine3d::Identity());
	
	/**
	 * @brief Constructor that takes the control frame in local link frame as a position and a rotation
	 * 
	 * @param robot          robot model
	 * @param link_name      link to which are attached the control frame and the sensor frame (end effector link)
	 * @param control_frame  Position of the control frame in link frame
	 * @param control_frame  Rotation of the control frame in link frame
	 */
	AllegroGrasp(RedisClient* redis_client,
					   Sai2Model::Sai2Model* robot,
					   const std::string link_name,
	                   const Eigen::Vector3d pos_in_link,
	                   const Eigen::Matrix3d rot_in_link = Eigen::Matrix3d::Identity());

	/**
	 * @brief destructor
	 */
	~AllegroGrasp();

	/**
	 * @brief Updates the primitive model (dynamic quantities for op space and kinematics of the control frame position). 
	 * Call it after calling the dunction updateModel of the robot model
	 */
	void updatePrimitiveModel();

	/**
	 * @brief Computes the joint torques associated with the primitive
	 * 
	 * @param torques   Vector that will be populated by the joint torques
	 */
	void computeTorques(Eigen::VectorXd& torques);

	void start(const Eigen::Vector3d obj_pos_base,
			   const Eigen::Matrix3d obj_rmat_base);

	void step();
	
	/**
	 * @brief Enable the gravity compensation at the primitive level (disabled by default)
	 * @details Use with robots that do not handle their own gravity compensation
	 */
	void enableGravComp();

	/**
	 * @brief disable the gravity compensation at the primitive level (default behavior)
	 * @details For robots that handle their own gravity compensation
	 */
	void disbleGravComp();

	RedisClient* _redis_client;
	Sai2Model::Sai2Model* _robot;
	std::string _link_name;
	Eigen::Affine3d _control_frame;

	RedundantArmMotion* _redundant_arm_motion;

	// state
	enum State {
		APPROACH, 
		PREGRASP,
		LOWER,
		GRASP,
		FORCE_GRASP,
		DONE
	};

	State _state;

	// counter
	int _iter;
	int _state_iter;

	// pick and place parameters
	Eigen::Vector3d _pos_pre_grasp;
	Eigen::Vector3d _pos_grasp;
	Eigen::Matrix3d _rot_obj;

protected:
	bool _gravity_compensation = false;

};


} /* namespace Sai2Primitives */

#endif /* SAI2_PRIMITIVES_ALLEGRO_GRASP_H_ */