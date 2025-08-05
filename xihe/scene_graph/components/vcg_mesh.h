#include <vcg/complex/complex.h>

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>

using namespace vcg;
using namespace tri;

class MyVertex;
class MyEdge;
class MyFace;

struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>::AsVertexType,
                                           vcg::Use<MyEdge>::AsEdgeType,
                                           vcg::Use<MyFace>::AsFaceType>
{};

class MyVertex : public vcg::Vertex<MyUsedTypes,
                                    vcg::vertex::Coord3f,
                                    vcg::vertex::Normal3f,
                                    vcg::vertex::TexCoord2f,
                                    vcg::vertex::Mark,
                                    vcg::vertex::BitFlags,
                                    vcg::vertex::Qualityf,
                                    vcg::vertex::VFAdj>
{
  public:
	vcg::math::Quadric<double> &Qd()
	{
		return q;
	}

  private:
	vcg::math::Quadric<double> q;
};

class MyEdge : public Edge<MyUsedTypes> {};

typedef BasicVertexPair<MyVertex> VertexPair;

class MyFace : public vcg::Face<MyUsedTypes,
                                vcg::face::VertexRef,
                                vcg::face::BitFlags,
                                vcg::face::VFAdj>
{};

class MyMesh : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace>>
{};

class MyTriEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric<MyMesh, VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>>
{
  public:
	typedef vcg::tri::TriEdgeCollapseQuadric<MyMesh, VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>> TECQ;
	typedef MyMesh::VertexType::EdgeType                                                                     EdgeType;
	inline MyTriEdgeCollapse(const VertexPair &p, int i, BaseParameterClass *pp) :
	    TECQ(p, i, pp)
	{}
};