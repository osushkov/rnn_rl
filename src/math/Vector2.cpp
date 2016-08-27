/*
 * Vector2.cpp
 *
 *  Created on: 05/12/2013
 *      Author: osushkov
 */

#include "Vector2.hpp"

#include <cmath>
#include <cassert>
#include <ostream>

static_assert(sizeof(Vector2) == 2*sizeof(double), "non-compact vector2");

Vector2::Vector2() :
    Vector2(0.0, 0.0) {}

Vector2::Vector2(double x, double y) :
    x(x), y(y) {}

Vector2::Vector2(const Vector2 &) = default;

void Vector2::set(double newX, double newY) {
  x = newX;
  y = newY;
}

void Vector2::normalise(void) {
  double invLength = 1.0 / length();
  assert(!isnan(invLength));

  x *= invLength;
  y *= invLength;
}

const Vector2 Vector2::normalised(void) const {
  Vector2 result(*this);
  result.normalise();
  return result;
}

double Vector2::dotProduct(const Vector2 &v) const {
  return (x * v.x) + (y * v.y);
}

double Vector2::absThetaTo(const Vector2 &v) const {
  double dp = dotProduct(v);
  if (dp > 1.0) {
    return 0.0;
  }
  if (dp < -1.0) {
    return M_PI;
  }

  return fabs(acos(dp));
}

double Vector2::length() const {
  return sqrt((x * x) + (y * y));
}

double Vector2::length2() const {
  return (x * x) + (y * y);
}

void Vector2::rotate(double theta) {
  double sinTheta = sin(theta);
  double cosTheta = cos(theta);

  double newX = (cosTheta * x) - (sinTheta * y);
  double newY = (sinTheta * x) + (cosTheta * y);

  x = newX;
  y = newY;
}

const Vector2 Vector2::rotated(double theta) const {
  Vector2 result(*this);
  result.rotate(theta);
  return result;
}

double Vector2::distanceTo(const Vector2 &v) const {
  return sqrt((x - v.x) * (x - v.x) + (y - v.y) * (y - v.y));
}

double Vector2::distanceTo2(const Vector2 &v) const {
  return (x - v.x) * (x - v.x) + (y - v.y) * (y - v.y);
}

void Vector2::reflect(const Vector2 &normal) {
  *this += normal * (-2.0 * normal.dotProduct(*this));
}

const Vector2 Vector2::reflected(const Vector2 &normal) const {
  Vector2 result(*this);
  result.reflect(normal);
  return result;
}

double Vector2::orientation() const {
  return atan2(y, x);
}

const Vector2 Vector2::operator+(const Vector2 &v) const {
  return Vector2(x + v.x, y + v.y);
}

const Vector2 Vector2::operator-(const Vector2 &v) const {
  return Vector2(x - v.x, y - v.y);
}

const Vector2 Vector2::operator*(double s) const {
  return Vector2(x * s, y * s);
}

const Vector2 Vector2::operator/(double s) const {
  return Vector2(x / s, y / s);
}

const Vector2& Vector2::operator+=(const Vector2 &v) {
  x += v.x;
  y += v.y;
  return *this;
}

const Vector2& Vector2::operator-=(const Vector2 &v) {
  x -= v.x;
  y -= v.y;
  return *this;
}

const Vector2& Vector2::operator*=(double s) {
  x *= s;
  y *= s;
  return *this;
}

const Vector2& Vector2::operator/=(double s) {
  x /= s;
  y /= s;
  return *this;
}

std::ostream& operator<<(std::ostream& stream, const Vector2& v) {
  stream << "(" << v.x << "," << v.y << ")";
  return stream;
}

const Vector2 operator*(double s, const Vector2 &v) {
  return Vector2(v.x * s, v.y * s);
}
