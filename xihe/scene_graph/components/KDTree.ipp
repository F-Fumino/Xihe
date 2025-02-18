#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include "KDTree.h"


inline float randomFloat(float min = 0.0f, float max = 1.0f)
{
	static std::random_device                     rd;
	static std::mt19937                           mt(rd());
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	return distribution(mt) * (max - min) + min;
}

namespace xihe::sg
{

#define KD_TREE_TEMPLATE template <typename TElement> requires HasSpatialInfo<TElement>

KD_TREE_TEMPLATE
KDTree<TElement>::KDTree()
{}

KD_TREE_TEMPLATE
KDTree<TElement>::KDTree(std::span<const TElement> elements)
{
	build(elements);
}

KD_TREE_TEMPLATE
void KDTree<TElement>::build(std::span<const TElement> elements)
{
	root         = {};
	elementCount = elements.size();
	std::vector<std::size_t> allPoints;
	allPoints.reserve(elements.size());
	for (std::size_t i = 0; i < elements.size(); i++)
	{
		allPoints.push_back(i);
	}
	const glm::vec3 regionMin{-INFINITY, -INFINITY, -INFINITY};
	const glm::vec3 regionMax{+INFINITY, +INFINITY, +INFINITY};

	buildInner(&root, elements, allPoints, 0, regionMin, regionMax);
}

KD_TREE_TEMPLATE
std::int64_t KDTree<TElement>::closestNeighbor(const TElement &from, float maxDistance) const
{
	Node *closest = findClosest(from);
	if (closest == nullptr)
	{
		return -1;
	}
	if (glm::distance2(from.getPosition(), closest->medianPoint) < maxDistance * maxDistance)
	{
		return closest->elementIndex;
	}
	return -1;
}

KD_TREE_TEMPLATE
void KDTree<TElement>::getNeighbors(std::vector<std::size_t> &out, const TElement &from, float maxDistance) const
{
	const glm::vec3 pos = from.getPosition();
	const glm::vec3 min = pos - glm::vec3(maxDistance);
	const glm::vec3 max = pos + glm::vec3(maxDistance);
	rangeSearch(out, min, max);
	// TODO: remove all > maxDistance (range search = hyperrectangle, neighbor search = hypersphere)
}

KD_TREE_TEMPLATE
void KDTree<TElement>::rangeSearch(std::vector<std::size_t> &out, const glm::vec3 &min, const glm::vec3 &max) const
{
	rangeSearchInner(out, &root, min, max);
}

KD_TREE_TEMPLATE
std::size_t KDTree<TElement>::size() const
{
	return this->elementCount;
}

KD_TREE_TEMPLATE
bool KDTree<TElement>::empty() const
{
	return size() == 0;
}

KD_TREE_TEMPLATE
void KDTree<TElement>::buildInner(Node *pDestination, std::span<const TElement> allElements, const std::vector<std::size_t> &subset, const std::size_t depth, const glm::vec3 &regionMin, const glm::vec3 &regionMax)
{
	assert(pDestination != nullptr);
	pDestination->regionMin = regionMin;
	pDestination->regionMax = regionMax;
	if (subset.size() == 1)
	{
		pDestination->elementIndex = *subset.begin();
		pDestination->medianPoint  = allElements[pDestination->elementIndex].getPosition();
		return;
	}
	if (subset.empty())
	{
		return;
	}

	const std::size_t axisIndex = depth % 3;

	// find median
	{
		std::vector<std::size_t> points;
		if (subset.size() < 512)
		{
			points.reserve(subset.size());
			for (std::size_t pointIndex : subset)
			{
				points.push_back(pointIndex);
			}
		}
		else
		{
			points.resize(512);
			for (std::size_t i = 0; i < 512; i++)
			{
				std::size_t randomIndex = randomFloat(0.0f, subset.size() - 1);
				points[i]               = subset[randomIndex];
			}
		}
		sort(points.begin(), points.end(), [&](const std::size_t &a, const std::size_t &b) {
			const float posA = allElements[a].getPosition()[axisIndex];
			const float posB = allElements[b].getPosition()[axisIndex];
			return posA < posB;
		});

		pDestination->elementIndex = points[points.size() / 2];
		pDestination->medianPoint  = allElements[pDestination->elementIndex].getPosition();
	}

	const float split = pDestination->medianPoint[axisIndex];

	std::vector<std::size_t> pointsBefore;
	std::vector<std::size_t> pointsAfter;

	for (std::size_t index : subset)
	{
		if (index == pDestination->elementIndex)
		{
			continue;        // already in tree
		}

		const glm::vec3 position = allElements[index].getPosition();
		if (position[axisIndex] < split)
		{
			pointsBefore.push_back(index);
		}
		else
		{
			pointsAfter.push_back(index);
		}
	}

	if (!pointsBefore.empty())
	{
		pDestination->pLeft          = std::make_unique<Node>();
		pDestination->pLeft->pParent = pDestination;
		glm::vec3 newMax             = regionMax;
		newMax[axisIndex]            = split;
		buildInner(pDestination->pLeft.get(), allElements, pointsBefore, depth + 1, regionMin, newMax);
	}
	if (!pointsAfter.empty())
	{
		auto ptr                      = std::make_unique<Node>();
		pDestination->pRight          = std::move(ptr);
		pDestination->pRight->pParent = pDestination;
		glm::vec3 newMin              = regionMin;
		newMin[axisIndex]             = split;
		buildInner(pDestination->pRight.get(), allElements, pointsAfter, depth + 1, newMin, regionMax);
	}
}

KD_TREE_TEMPLATE
typename KDTree<TElement>::Node *KDTree<TElement>::findClosest(const TElement &from) const
{
	if (empty())
	{
		return nullptr;
	}

	// 1. find leaf with the closest element
	Node       *currentNode = &root;
	std::size_t depth       = 0;

	const float nodePositions[3] = {
	    from.getPosition()[0],
	    from.getPosition()[1],
	    from.getPosition()[2],
	};

	const std::size_t maxIterations = size();
	std::size_t       iteration     = 0;
	bool              found         = false;
	while (iteration++ < maxIterations)
	{
		const std::size_t axisIndex = depth % 3;
		depth++;

		const float split = currentNode->medianPoint[axisIndex];
		if (nodePositions[axisIndex] < split)
		{
			Node *nextNode = currentNode->pLeft.get();
			if (nextNode == nullptr)
			{
				found = true;
				break;        // closest is 'currentNode'
			}
			else
			{
				currentNode = nextNode;
			}
		}
		else
		{
			Node *nextNode = currentNode->pRight.get();
			if (nextNode == nullptr)
			{
				found = true;
				break;        // closest is 'currentNode'
			}
			else
			{
				currentNode = nextNode;
			}
		}
	}

	assert(found);

	return currentNode;
}

KD_TREE_TEMPLATE
void KDTree<TElement>::rangeSearchInner(std::vector<std::size_t> &out, const Node *pRoot, const glm::vec3 &min, const glm::vec3 &max) const
{
	const bool isLeaf = pRoot->pLeft == nullptr && pRoot->pRight == nullptr;
	if (isLeaf)
	{
		if (glm::all(glm::greaterThanEqual(pRoot->medianPoint, min)) && glm::all(glm::lessThan(pRoot->medianPoint, max)))
		{
			out.push_back(pRoot->elementIndex);
		}
	}
	else
	{
		auto regionFullyContained = [&](const Node &node) -> bool {
			return glm::all(glm::greaterThanEqual(node.regionMin, min)) && glm::all(glm::lessThan(node.regionMax, max));
		};
		auto regionIntersects = [&](const Node &node) -> bool {
			return glm::all(glm::lessThanEqual(node.regionMin, max)) && glm::all(glm::greaterThan(node.regionMax, min));
		};

		std::function<void(const Node &)> reportSubTree = [&](const Node &node) {
			out.push_back(node.elementIndex);
			if (node.pLeft)
			{
				reportSubTree(*node.pLeft);
			}
			if (node.pRight)
			{
				reportSubTree(*node.pRight);
			}
		};

		if (pRoot->pLeft)
		{
			if (regionFullyContained(*pRoot->pLeft))
			{
				reportSubTree(*pRoot->pLeft);
			}
			else if (regionIntersects(*pRoot->pLeft))
			{
				rangeSearchInner(out, pRoot->pLeft.get(), min, max);
			}
		}
		if (pRoot->pRight)
		{
			if (regionFullyContained(*pRoot->pRight))
			{
				reportSubTree(*pRoot->pRight);
			}
			else if (regionIntersects(*pRoot->pRight))
			{
				rangeSearchInner(out, pRoot->pRight.get(), min, max);
			}
		}
	}
}
}        // namespace xihe::sg