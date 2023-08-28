#include "POPCExplicitForceControl.h"

namespace Sai2Primitives {

POPCExplicitForceControl::POPCExplicitForceControl(const double loop_timestep)
	: _loop_timestep(loop_timestep), _is_enabled(false) {
	reInitialize();
}

void POPCExplicitForceControl::reInitialize() {
	_passivity_observer_value = 0;
	_E_correction = 0;
	_stored_energy_PO = 0;
	std::queue<double> empty;
	std::swap(_PO_buffer_window, empty);

	_PO_counter = _PO_max_counter;

	_Rc = 1.0;
	_vcl_squared_sum = 0;
}

void POPCExplicitForceControl::enable() { _is_enabled = true; }

void POPCExplicitForceControl::disable() {
	_is_enabled = false;
	reInitialize();
}

Vector3d POPCExplicitForceControl::computePassivitySaturatedForce(
	const Vector3d& fd, const Vector3d& fs, const Vector3d& vcl,
	const Vector3d& vr, const Matrix3d kv_force, const double k_feedforward) {
	if (!_is_enabled) {
		return vcl - kv_force * vr;
	}

	Vector3d F_cmd = k_feedforward * fd + _Rc * vcl - kv_force * vr;
	double vc_squared = vcl.dot(vcl);
	Vector3d f_diff = fs - fd;

	// compute power input and output
	double power_input_output =
		(f_diff.dot(vcl) - F_cmd.dot(vr)) * _loop_timestep;

	// windowed PO
	_passivity_observer_value += power_input_output;
	_PO_buffer_window.push(power_input_output);

	if (_passivity_observer_value + _stored_energy_PO + _E_correction > 0) {
		while (_PO_buffer_window.size() > _PO_window_size) {
			if (_passivity_observer_value + _E_correction + _stored_energy_PO >
				_PO_buffer_window.front()) {
				if (_PO_buffer_window.front() > 0) {
					_passivity_observer_value -= _PO_buffer_window.front();
				}
				_PO_buffer_window.pop();
			} else {
				break;
			}
		}
	}

	// compute PC
	if (_PO_counter <= 0) {
		_PO_counter = _PO_max_counter;

		double old_Rc = _Rc;
		if (_passivity_observer_value + _stored_energy_PO + _E_correction <
			0)	// activity detected
		{
			_Rc = 1 + (_passivity_observer_value + _stored_energy_PO +
					   _E_correction) /
						  (_vcl_squared_sum * _loop_timestep);

			if (_Rc > 1) {
				_Rc = 1;
			}
			if (_Rc < 0) {
				_Rc = 0;
			}

		} else	// no activity detected
		{
			_Rc = (1 + (0.1 * _PO_max_counter - 1) * _Rc) /
				  (double)(0.1 * _PO_max_counter);
		}

		_E_correction += (1 - old_Rc) * _vcl_squared_sum * _loop_timestep;
		_vcl_squared_sum = 0;
	}

	_PO_counter--;
	_vcl_squared_sum += vc_squared;

	return _Rc * vcl - kv_force * vr;
}

}  // namespace Sai2Primitives
