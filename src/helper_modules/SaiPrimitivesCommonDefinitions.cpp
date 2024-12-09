#include "SaiPrimitivesCommonDefinitions.h"

using namespace Eigen;

namespace SaiPrimitives {

VectorXd extractKpFromGainVector(const std::vector<PIDGains>& gains) {
	VectorXd kp(gains.size());
	for (int i = 0; i < gains.size(); ++i) {
		kp(i) = gains[i].kp;
	}
	return kp;
}

VectorXd extractKvFromGainVector(const std::vector<PIDGains>& gains) {
	VectorXd kv(gains.size());
	for (int i = 0; i < gains.size(); ++i) {
		kv(i) = gains[i].kv;
	}
	return kv;
}

VectorXd extractKiFromGainVector(const std::vector<PIDGains>& gains) {
	VectorXd ki(gains.size());
	for (int i = 0; i < gains.size(); ++i) {
		ki(i) = gains[i].ki;
	}
	return ki;
}

}  // namespace SaiPrimitives