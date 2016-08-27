/*
 *
 *  Created on: 04/12/2013
 *      Author: osushkov
 */

#pragma once

#include <iosfwd>

class Vector2 final {
public:

  double x;
  double y;

  Vector2();
  Vector2(double x, double y);
  Vector2(const Vector2 &);

  void set(double newX, double newY);

  void normalise(void);
  const Vector2 normalised(void) const;

  double dotProduct(const Vector2 &v) const;
  double absThetaTo(const Vector2 &v) const;

  double length() const;
  double length2() const;

  void rotate(double theta);
  const Vector2 rotated(double theta) const;

  double distanceTo(const Vector2 &v) const;
  double distanceTo2(const Vector2 &v) const;

  void reflect(const Vector2 &normal);
  const Vector2 reflected(const Vector2 &normal) const;

  double orientation() const;

  const Vector2 operator+(const Vector2 &v) const;
  const Vector2 operator-(const Vector2 &v) const;
  const Vector2 operator*(double s) const;
  const Vector2 operator/(double s) const;

  const Vector2& operator+=(const Vector2 &v);
  const Vector2& operator-=(const Vector2 &v);
  const Vector2& operator*=(double s);
  const Vector2& operator/=(double s);
};

std::ostream& operator<<(std::ostream& stream, const Vector2& v);
const Vector2 operator*(double s, const Vector2 &v);
