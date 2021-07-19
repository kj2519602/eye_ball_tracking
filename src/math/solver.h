#ifndef _SRC_MATH_SOLVER_H_
#define _SRC_MATH_SOLVER_H_

#include <cmath>

#include "math/geometry.h"

namespace pupil {

// ax + b = 0
Scalar solve(Scalar a, Scalar b);

// ax^2 + bx + c = 0
Vector2 solver(Scalar a, Scalar b, Scalar c);

// ax^2 + bx + c = 0
Vector3 solver(Scalar a, Scalar b, Scalar c, Scalar d);

}  // namespace pupil

#endif//_SRC_MATH_SOLVER_H_
