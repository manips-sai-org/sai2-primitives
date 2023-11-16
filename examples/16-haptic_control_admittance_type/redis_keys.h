#include <string>

using namespace std;

const string REDIS_KEY_PREFIX = "sai2::ChaiHapticDevice::device0";

const string MAX_STIFFNESS_KEY =
	REDIS_KEY_PREFIX + "::specifications::max_stiffness";
const string MAX_DAMPING_KEY =
	REDIS_KEY_PREFIX + "::specifications::max_damping";
const string MAX_FORCE_KEY = REDIS_KEY_PREFIX + "::specifications::max_force";
const string COMMANDED_FORCE_KEY =
	REDIS_KEY_PREFIX + "::actuators::commanded_force";
const string COMMANDED_TORQUE_KEY =
	REDIS_KEY_PREFIX + "::actuators::commanded_torque";
const string COMMANDED_GRIPPER_FORCE_KEY =
	REDIS_KEY_PREFIX + "::actuators::commanded_force_gripper";
const string POSITION_KEY = REDIS_KEY_PREFIX + "::sensors::current_position";
const string ROTATION_KEY = REDIS_KEY_PREFIX + "::sensors::current_rotation";
const string GRIPPER_POSITION_KEY =
	REDIS_KEY_PREFIX + "::sensors::current_position_gripper";
const string LINEAR_VELOCITY_KEY =
	REDIS_KEY_PREFIX + "::sensors::current_trans_velocity";
const string ANGULAR_VELOCITY_KEY =
	REDIS_KEY_PREFIX + "::sensors::current_rot_velocity";
const string GRIPPER_VELOCITY_KEY =
	REDIS_KEY_PREFIX + "::sensors::current_gripper_velocity";
const string SENSED_FORCE_KEY = REDIS_KEY_PREFIX + "::sensors::sensed_force";
const string SENSED_TORQUE_KEY = REDIS_KEY_PREFIX + "::sensors::sensed_torque";
const string USE_GRIPPER_AS_SWITCH_KEY =
	REDIS_KEY_PREFIX + "::sensors::use_gripper_as_switch";
const string SWITCH_PRESSED_KEY = REDIS_KEY_PREFIX + "::sensors::switch_pressed";
