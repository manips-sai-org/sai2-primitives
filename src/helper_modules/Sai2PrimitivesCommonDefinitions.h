#ifndef SAI2_PRIMITIVES_COMMON_DEFINITIONS_H_
#define SAI2_PRIMITIVES_COMMON_DEFINITIONS_H_

#include <Eigen/Dense>
#include <vector>

namespace Sai2Primitives {

enum DynamicDecouplingType {
	FULL_DYNAMIC_DECOUPLING,	// use the real Mass matrix
	BOUNDED_INERTIA_ESTIMATES,	// use a Mass matrix computed from
								// saturating the minimal values of the Mass
								// Matrix
	IMPEDANCE,					// use Identity for the Mass matrix
};

struct PIDGains {
	double kp;
	double kv;
	double ki;

	PIDGains(double kp, double kv, double ki) : kp(kp), kv(kv), ki(ki) {}
};

Eigen::VectorXd extractKpFromGainVector(const std::vector<PIDGains>& gains);
Eigen::VectorXd extractKvFromGainVector(const std::vector<PIDGains>& gains);
Eigen::VectorXd extractKiFromGainVector(const std::vector<PIDGains>& gains);

}  // namespace Sai2Primitives

#endif	// SAI2_PRIMITIVES_COMMON_DEFINITIONS_H_
