from strike_a_pose.datagen.skeleton import NUM_JOINTS, SMPL24_JOINT_NAMES, SMPL24_PARENTS


def test_skeleton_definition():
    assert NUM_JOINTS == 24
    assert len(set(SMPL24_JOINT_NAMES)) == 24
    assert SMPL24_PARENTS[0] == -1
    assert all(0 <= p < i for i, p in enumerate(SMPL24_PARENTS) if i > 0)
    assert SMPL24_JOINT_NAMES[0] == "pelvis" and SMPL24_JOINT_NAMES[15] == "head"
