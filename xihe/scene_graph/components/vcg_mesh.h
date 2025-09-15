#include <vcg/complex/complex.h>

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/edge_collapse.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>
#include <vcg/container/simple_temporary_data.h>

class MyVertex;
class MyEdge;
class MyFace;

struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>::AsVertexType,
                                           vcg::Use<MyEdge>::AsEdgeType,
                                           vcg::Use<MyFace>::AsFaceType>
{
};

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

class MyEdge : public vcg::Edge<MyUsedTypes>
{};

typedef vcg::tri::BasicVertexPair<MyVertex> VertexPair;

class MyFace : public vcg::Face<MyUsedTypes,
                                vcg::face::VertexRef,
                                vcg::face::Normal3f,
                                vcg::face::BitFlags,
								vcg::face::FFAdj,
                                vcg::face::VFAdj,
								vcg::face::Mark>
{
};

class MyMesh : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace>>
{
};

uint32_t CountConnectedComponents(MyMesh &mesh)
{
	vcg::tri::UpdateTopology<MyMesh>::FaceFace(mesh);
	vcg::tri::RequirePerFaceMark(mesh);
	vcg::tri::UnMarkAll(mesh);

	vcg::tri::ConnectedComponentIterator<MyMesh> cci;
	uint32_t                                     componentCount = 0;

	for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi)
	{
		if (!vcg::tri::IsMarked(mesh, &(*fi)))
		{
			++componentCount;
			cci.start(mesh, &(*fi));
			while (!cci.completed())
			{
				auto fp = *cci;
				++cci;
			}
		}
	}
	return componentCount;
}

class MyTriEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric<MyMesh, VertexPair, MyTriEdgeCollapse, vcg::tri::QInfoStandard<MyVertex>>
{
  public:
	typedef vcg::tri::TriEdgeCollapseQuadric<MyMesh, VertexPair, MyTriEdgeCollapse, vcg::tri::QInfoStandard<MyVertex>> TECQ;
	typedef MyMesh::VertexType::EdgeType                                                                     EdgeType;
	inline MyTriEdgeCollapse(const VertexPair &p, int i, vcg::BaseParameterClass *pp) :
	    TECQ(p, i, pp)
	{}
};


//namespace vcg
//{
//namespace tri
//{
//
//typedef SimpleTempData<MyMesh::VertContainer, math::Quadric<double>> QuadricTemp;
//
//class QHelper
//{
//  public:
//	QHelper()
//	{}
//	static void Init()
//	{}
//	static math::Quadric<double> &Qd(MyVertex &v)
//	{
//		return TD()[v];
//	}
//	static math::Quadric<double> &Qd(MyVertex *v)
//	{
//		return TD()[*v];
//	}
//	static MyVertex::ScalarType W(MyVertex * /*v*/)
//	{
//		return 1.0;
//	}
//	static MyVertex::ScalarType W(MyVertex & /*v*/)
//	{
//		return 1.0;
//	}
//	static void Merge(MyVertex & /*v_dest*/, MyVertex const & /*v_del*/)
//	{}
//	static QuadricTemp *&TDp()
//	{
//		static QuadricTemp *td;
//		return td;
//	}
//	static QuadricTemp &TD()
//	{
//		return *TDp();
//	}
//};
//
//typedef BasicVertexPair<MyVertex> VertexPair;
//
//class MyTriEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric<MyMesh, VertexPair, MyTriEdgeCollapse, QHelper>
//{
//  public:
//	typedef vcg::tri::TriEdgeCollapseQuadric<MyMesh, VertexPair, MyTriEdgeCollapse, QHelper> TECQ;
//	inline MyTriEdgeCollapse(const VertexPair &p, int i, BaseParameterClass *pp) :
//	    TECQ(p, i, pp)
//	{}
//};
//
//}        // end namespace tri
//}        // end namespace vcg