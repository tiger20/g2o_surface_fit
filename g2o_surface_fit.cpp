#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>
#include <iostream>

#include "g2o/stuff/sampler.h"
#include "g2o/stuff/command_args.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

vector<Point3f> generatePoints()
{
    int numPts = 1000;
    vector<Point3f> vPt;
    RNG rng;
    double w_sigma = 0.5;

    for(int i = 0;i < numPts;++i)
    {
        Point3f pt;
        pt.x = rng.uniform(0.f,5.f);
        pt.y = rng.uniform(0.f,5.f);
        pt.z = (-2)*pt.x-3*pt.y+2;
        pt.z += rng.gaussian(w_sigma);

        vPt.push_back(pt);
    }

    return vPt;
}

class VertexParams : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexParams()
    {
    }

    virtual bool read(std::istream& /*is*/)
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    virtual bool write(std::ostream& /*os*/) const
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    virtual void setToOriginImpl()
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
    }

    virtual void oplusImpl(const double* update)
    {
        Eigen::Vector3d::ConstMapType v(update);
        _estimate += v;
    }
};

class EdgePointOnSurface : public g2o::BaseUnaryEdge<1, Eigen::Vector3d, VertexParams>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePointOnSurface()
    {
    }
    virtual bool read(std::istream& /*is*/)
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
    virtual bool write(std::ostream& /*os*/) const
    {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    void computeError()
    {
        const VertexParams* params = static_cast<const VertexParams*>(vertex(0));
        const double& a = params->estimate()(0);
        const double& b = params->estimate()(1);
        const double& c = params->estimate()(2);
        //const double& d = params->estimate()(3);
        //double fval = a * exp(-lambda * measurement()(0)) + b;
        _error(0) = measurement()(2) - (a * measurement()(0) + b * measurement()(1) + c);
    }
};

int main()
{
    vector<Point3f> cloud;
    cloud = generatePoints();
    bool verbose = true;
    int maxIterations = 20;
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");

    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    /// Let's assume camera has the following properties
    Vec3f cam_pos(3.0f,3.0f,3.0f), cam_focal_point(3.0f,3.0f,2.0f), cam_y_dir(-1.0f,0.0f,0.0f);

    /// We can get the pose of the cam using makeCameraPose
    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

    /// We can get the transformation matrix from camera coordinate system to global using
    /// - makeTransformToGlobal. We need the axes of the camera
    Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f,-1.0f,0.0f), Vec3f(-1.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,-1.0f), cam_pos);

    viz::WCloud cloud_widget(cloud, viz::Color::green());

    /// Pose of the widget in camera frame
    Affine3f cloud_pose = Affine3f().translate(Vec3f(0.0f,0.0f,0.0f));
    /// Pose of the widget in global frame
    Affine3f cloud_pose_global = transform * cloud_pose;

    myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);


    typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
    typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyLinearSolver;
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<MyBlockSolver>(g2o::make_unique<MyLinearSolver>()));

    optimizer.setAlgorithm(solver);

    // build the optimization problem given the points
    // 1. add the parameter vertex
    VertexParams* params = new VertexParams();
    params->setId(0);
    params->setEstimate(Eigen::Vector3d(-2,-2,1)); // some initial value for the params
    optimizer.addVertex(params);
    // 2. add the points we measured to be on the curve

    for(int i = 0;i < cloud.size();++i)
    {
        EdgePointOnSurface* e = new EdgePointOnSurface;
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        e->setVertex(0, params);
        Eigen::Vector3d mPoint;
        mPoint[0] = cloud[i].x;
        mPoint[1] = cloud[i].y;
        mPoint[2] = cloud[i].z;
        e->setMeasurement(mPoint);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.setVerbose(verbose);
    optimizer.optimize(maxIterations);

    cout<<params->estimate()(0)<<endl;
    cout<<params->estimate()(1)<<endl;
    cout<<params->estimate()(2)<<endl;
    //cout<<params->estimate()(3)<<endl;

    //show plane

    Point3d pCenter;
    pCenter.x = 2.5;
    pCenter.y = 2.5;
    pCenter.z = params->estimate()(0) * 2.5 + params->estimate()(1) * 2.5 + params->estimate()(2);
    Vec3d norm;
    norm[0] = (-1) * params->estimate()(0);
    norm[1] = (-1) * params->estimate()(1);
    norm[2] = 1;

    viz::WPlane plane(pCenter,norm,Vec3d(0,1,0),Size(10,10));

    myWindow.showWidget("plane", plane, cloud_pose_global);
    myWindow.setRenderingProperty("plane",cv::viz::OPACITY,0.5);

    myWindow.spin();


    return 0;

}
