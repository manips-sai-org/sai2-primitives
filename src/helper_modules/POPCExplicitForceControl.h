#ifndef SAI2_PRIMITIVES_POPCEXPLICITFORCECONTROL_TASK_H_
#define SAI2_PRIMITIVES_POPCEXPLICITFORCECONTROL_TASK_H_

#include <Eigen/Dense>
#include <queue>

using namespace Eigen;
using namespace std;

namespace Sai2Primitives {

class POPCExplicitForceControl {
public:
	POPCExplicitForceControl(const double loop_timestep);
	~POPCExplicitForceControl() = default;

	// disallow empty, copy and asssign constructors
	POPCExplicitForceControl() = delete;
	POPCExplicitForceControl(POPCExplicitForceControl const&) = delete;
	POPCExplicitForceControl& operator=(POPCExplicitForceControl const&) =
		delete;

    void reInitialize();
    void enable();
    void disable();

	Vector3d computePassivitySaturatedForce(
		const Vector3d& fd, const Vector3d& fs,
		const Vector3d& vcl, const Vector3d& vr,
		const Matrix3d kv_force = Matrix3d::Zero(), const double k_feedforward = 0);

private:

    double _loop_timestep;
	bool _is_enabled;

	const int _PO_window_size = 250;
	const int _PO_max_counter = 50;

    Matrix3d _sigma_force;

	double _passivity_observer_value;
	double _E_correction;
	double _stored_energy_PO;
	queue<double> _PO_buffer_window;

	int _PO_counter;

    double _vcl_squared_sum;
	double _Rc;
};

}  // namespace Sai2Primitives

#endif	// SAI2_PRIMITIVES_POPCEXPLICITFORCECONTROL_TASK_H_