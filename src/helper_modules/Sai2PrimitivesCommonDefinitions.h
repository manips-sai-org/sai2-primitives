#ifndef SAI2_PRIMITIVES_COMMON_DEFINITIONS_H_
#define SAI2_PRIMITIVES_COMMON_DEFINITIONS_H_

namespace Sai2Primitives
{

struct PIDGains {
	double kp;
	double kv;
	double ki;

	PIDGains(double kp, double kv, double ki) : kp(kp), kv(kv), ki(ki) {}
};

}

#endif // SAI2_PRIMITIVES_COMMON_DEFINITIONS_H_
