// SurfaceProjectorManager.h

#pragma once
#include "MathUtils.h"
#include <queue>
#include <mutex>

namespace cloud
{
    namespace accumulator
    {

        enum class ProjectorMessageType
        {
            SetNewTarget,
            RecomputeTarget,
            UpdateTarget,
            NewVolumeID,
            VolumeCleanup,
            Exit
        };

        // surface projector worker class
        class SurfaceProjectorManager
        {

            // Member Variables
            // ----------------
        protected:

            // mapped surface target
            bool m_mappedTargSet;
            std::vector<Point3f> m_mappedSurfPositions;
            Point3f m_mappedSurfClickedPos;
            Vector3f m_mappedSurfNormal;

            // voxel removal data
            Matrix4f m_tipTransform;

            int m_updatedVolumeID;

            std::vector<Point3f> m_newlyAddedVoxels;

            // message queue variables
            std::queue<ProjectorMessageType> m_msgQueue;
            std::mutex m_projectorMutex;
            std::condition_variable m_conditionVarQueueEmpty;


            // Constructor/Destructor
            // ----------------------
        public:
            SurfaceProjectorManager();
            ~SurfaceProjectorManager();


            // Public Methods
            // --------------
        public:
            // UI / other thread input methods
            void SetMappedClickData(std::vector<Point3f> && positionsIn, const Point3f & clickedPosIn, const Vector3f & normIn);
            void SetNavigationFailTransform(const Matrix4f& transformMat);
            void UpdateVolumeID(int volID);
            void AddNewVoxels(std::vector<Point3f> && newVoxels);
            void ClearTarget();

            // getters/takers
            bool MappedTargetSet();
            int GetNewVolumeID();
            std::vector<Point3f> TakeNewlyAddedVoxels();
            void GetMappedData(std::vector<Point3f> & localTriangles, Point3f &hitPos, Vector3f &hitNorm);
            void GetNavigationFailTransform(Matrix4f& transformMat);

            void AddWorkMessage(ProjectorMessageType messageIn);

            ProjectorMessageType GetNextWorkItem();
        };

    }
}
