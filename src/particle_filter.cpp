/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

#define DEFAULT_NUMBER_OF_PARTICLES (50)
#define MIN_VALUE (0.0000001)

using std::string;
using std::vector;
using std::normal_distribution;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{

  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  /* Define the number of particles  */
  num_particles = DEFAULT_NUMBER_OF_PARTICLES;
  /* Define the normal distributions of x, y and theta information  */
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  /* Define particles as required and append them to particles list */
  for (uint8_t i_u8 = 0U; i_u8 < num_particles; ++i_u8)
  {
    Particle particle;

    particle.id = i_u8;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1U;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  /* Set the flag indicating that particles have been initialized */
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

    double x;
    double y;
    double theta;

    /* Define the normal distributions for positions and direction */
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (uint8_t i_u8 = 0U; i_u8 < num_particles; i_u8++)
    {
        x = particles[i_u8].x;
        y = particles[i_u8].y;
        theta = particles[i_u8].theta;

        /* Calculate the position of the particles if the path of the vehicle is circular */
        if (fabs(yaw_rate) < MIN_VALUE)
        {
            x += velocity * delta_t * cos(theta);
            y += velocity * delta_t * sin(theta);
        }
        /* Calculate the position of the particles if the path of the vehicle is straight */
        else
        {
            x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
            y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
            theta += yaw_rate * delta_t;
        }

        /* Predict the position of the particle */
        particles[i_u8].x = x + dist_x(gen);
        particles[i_u8].y = y + dist_y(gen);
        particles[i_u8].theta = theta + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

   double current_dist;
   double closest_dist;

  /* Get for each measurement the prediction with the closest position */
  for (unsigned int i_u8 = 0U; i_u8 < observations.size(); i_u8++)
  {
      /* Current closest dist between prediction and measurement; initialized to max value of double data type */
      closest_dist = std::numeric_limits<double>::max();
      /* Check for each prediction the distance to the current measurement */
      for (unsigned int j_u8 = 0U; j_u8 < predicted.size(); j_u8++)
      {
          current_dist = dist(observations[i_u8].x, observations[i_u8].y, predicted[j_u8].x, predicted[j_u8].y);
          /* Update measurement information regarding to closest prediction */
          if (current_dist < closest_dist)
          {
              closest_dist = current_dist;
              observations[i_u8].id = predicted[j_u8].id;
          }
      }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   LandmarkObs transfObservation;
   double x_part;
   double y_part;
   double theta_part;
   double landmark_dist_square;
   double sensor_range_square;
   double x_delta;
   double y_delta;
   double gauss_norm;
   double gauss_exponent;

   double x_obs;
   double y_obs;
   double mu_x;
   double mu_y;

   double sig_x = std_landmark[0];
   double sig_y = std_landmark[1];

   double weight_sum = 0;

   unsigned int p_u8;
   unsigned int i_u8;
   unsigned int j_u8;

   gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

   for (p_u8 = 0; p_u8 < num_particles; p_u8++)
   {
       x_part = particles[p_u8].x;
       y_part = particles[p_u8].y;
       theta_part = particles[p_u8].theta;
       vector<LandmarkObs> relevantLandmarks;

       for (i_u8 = 0U; i_u8 < map_landmarks.landmark_list.size(); i_u8++)
       {
           x_delta = map_landmarks.landmark_list[i_u8].x_f - x_part;
           y_delta = map_landmarks.landmark_list[i_u8].y_f - y_part;
           landmark_dist_square = x_delta * x_delta + y_delta * y_delta;
           sensor_range_square = sensor_range * sensor_range;
           /* Only consider landmarks which are within the sensor range */
           if (landmark_dist_square <= sensor_range_square)
           {
               relevantLandmarks.push_back(LandmarkObs{map_landmarks.landmark_list[i_u8].id_i,
                                            map_landmarks.landmark_list[i_u8].x_f,
                                            map_landmarks.landmark_list[i_u8].y_f});
           }
       }
       /* Transform observations in Map Coordinates */
       vector<LandmarkObs> observationsInMapCoordinates;

       for (i_u8 = 0; i_u8 < observations.size(); ++i_u8)
       {
           transfObservation.id = observations[i_u8].id;
           transfObservation.x = x_part + (cos(theta_part) * observations[i_u8].x) -
                                           (sin(theta_part) * observations[i_u8].y);
           transfObservation.y = y_part + (sin(theta_part) * observations[i_u8].x) +
                                           (cos(theta_part) * observations[i_u8].y);
           observationsInMapCoordinates.push_back(transfObservation);
       }

       dataAssociation(relevantLandmarks ,observationsInMapCoordinates);

       /* Init weights to calculate new gaussian distributed value properly */
       particles[p_u8].weight = 1.0;
       weights[p_u8] = 1.0;

       /* Check for the associated landmark for each observation and calculate the updated weight value */
       for (i_u8 = 0U; i_u8 < observationsInMapCoordinates.size(); ++i_u8)
       {
           x_obs = observationsInMapCoordinates[i_u8].x;
           y_obs = observationsInMapCoordinates[i_u8].y;

           for (j_u8 = 0U; j_u8 < relevantLandmarks.size(); j_u8++)
           {
               if (relevantLandmarks[j_u8].id == observationsInMapCoordinates[i_u8].id)
               {
                   mu_x = relevantLandmarks[j_u8].x;
                   mu_y = relevantLandmarks[j_u8].y;
                   break;
               }
           }

           /* Update particle weight using a mult-variate Gaussian */
           gauss_exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                            + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
           /* Final weight is calculated over all observations */
           particles[p_u8].weight *= gauss_norm * exp(-gauss_exponent);
       }

       /* Fill weights list */
       weights[p_u8] = particles[p_u8].weight;
       weight_sum += weights[p_u8];
   }
   /* Normalize weights */

   if (fabs(weight_sum) > 0.0)
   {
       for (p_u8 = 0U; p_u8 < weights.size(); p_u8++)
       {
           weights[p_u8] = weights[p_u8] / weight_sum;
       }
   }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> newParticles;
  std::discrete_distribution<int> dd(weights.begin(), weights.end());
  uint8_t i_u8;
  uint8_t newParticleIndex;

  for (i_u8 = 0U; i_u8 < num_particles; i_u8++)
  {
      newParticleIndex= dd(gen);
      newParticles.push_back(particles[newParticleIndex]);
  }

  particles = newParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
