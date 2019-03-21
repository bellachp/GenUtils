// SurfaceProjectorManager.cpp


#include "stdafx.h"
#include "SurfaceProjectorManager.h"


using namespace cloud::accumulator;



SurfaceProjectorManager::SurfaceProjectorManager()
{
    m_mappedTargSet = false;
}

SurfaceProjectorManager::~SurfaceProjectorManager()
{
}

// UI / input methods
void SurfaceProjectorManager::SetMappedClickData(std::vector<Point3f> && trisIn, const Point3f & clickIn, const Vector3f & normIn)
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);
        
    m_mappedSurfClickedPos = clickIn;
    m_mappedSurfNormal = normIn;      
    m_mappedSurfPositions = std::move(trisIn);
    m_mappedTargSet = true;
}

void SurfaceProjectorManager::SetNavigationFailTransform(const Matrix4f& transformMat)
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);

    m_tipTransform = transformMat;
}

void SurfaceProjectorManager::UpdateVolumeID(int volID)
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);
    m_updatedVolumeID = volID;
    m_msgQueue.push(ProjectorMessageType::NewVolumeID);
    m_conditionVarQueueEmpty.notify_one();
}

void SurfaceProjectorManager::AddNewVoxels(std::vector<Point3f> && newVoxels)
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);
    for (auto & pt : newVoxels)
    {
        m_newlyAddedVoxels.push_back(pt);
    }
}

void SurfaceProjectorManager::ClearTarget()
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);
    m_mappedTargSet = false;
}


// getters for worker/ui
bool SurfaceProjectorManager::MappedTargetSet()
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);
    return m_mappedTargSet;
}

int SurfaceProjectorManager::GetNewVolumeID()
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);
    return m_updatedVolumeID;
}

std::vector<Point3f> SurfaceProjectorManager::TakeNewlyAddedVoxels()
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);
    auto takenVoxels = std::move(m_newlyAddedVoxels);
    m_newlyAddedVoxels.clear(); // returns moved vector to valid state
    return takenVoxels;
}

void SurfaceProjectorManager::GetMappedData(std::vector<Point3f> & localTriangles, Point3f &hitPos, Vector3f &hitNorm)
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);

    localTriangles = std::move(m_mappedSurfPositions);
    hitPos = m_mappedSurfClickedPos;
    hitNorm = m_mappedSurfNormal;
    m_mappedSurfPositions.clear(); // returns moved vector to valid state
}

void SurfaceProjectorManager::GetNavigationFailTransform(Matrix4f& transformMat)
{
    std::lock_guard<std::mutex> guard(m_projectorMutex);

    transformMat = m_tipTransform;
}


// queue methods
void SurfaceProjectorManager::AddWorkMessage(ProjectorMessageType messageIn)
{
    std::unique_lock<std::mutex> ulock(m_projectorMutex);

    m_msgQueue.push(messageIn);
    m_conditionVarQueueEmpty.notify_one();
}

ProjectorMessageType SurfaceProjectorManager::GetNextWorkItem()
{
    std::unique_lock<std::mutex> ulock(m_projectorMutex);
    while (m_msgQueue.empty())
    {
        m_conditionVarQueueEmpty.wait(ulock);
    }

    ProjectorMessageType workItem = m_msgQueue.front();
    m_msgQueue.pop();
    return workItem;
}
