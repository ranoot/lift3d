# Creating a new, simpler interface:

Currenty, all the running logic lies in run_semantic_universe, however, the way to set it up and run it involves the use of hooks due to the long times associated with running the full program. This greatly overcomplicates installation and deployment on the internal robot dog system. It be much easier to use an interface similar to the onsync function present in online semantic, we do not care if the function is blocking, so just have a function "output runOnSyncedInput(input)". 

I will now describe to you the actual interface required.

```C++
struct OpenVocabSegInput{
    std::vector<Common::Entity::Image> images;
    Common::Entity::PointCloud pointCloud;
    Common::Entity::PrimaryPose pose;
};
// Some explanation regarding the input, you can assume that all of these are timesynced (and interpolated), now std::vector will only actually contain 1 image, don't forget to add some sort of error checking to make sure only got 1.

// Return time-synced image and point cloud. Pose is interpolated to align exactly 
// with image timestamp.
std::optional<OpenVocabSegInput> getSynchronizedInputs();

// Our function should be (you may choose a suitable name for it)
std::vector<Common::Entity::EAIRoomObject> runOnInput( std::optional<OpenVocabSegInput> input );
// It would make sense to not do anything if the optional is null
// You can output the entire object list here. Maybe just try not to run the whole object input convex polygon algorithm for all objects everytime lol
```

You can hopefully package this as a functor? I would like it to encapsulate all the interfaces to run the entire pipeline, whether that be onlinesemantic semanticvocab ... (everything else)

## Preserving logging and live viewing capabilities:

Currently our visualizer and pipeline logic are separate, as you have done in a previous step. This new runOnInput function should also be publishing to the IPC endpoint for the python rerun server to hopefully read. It should also write to an .rrd file on call, and log to stdout.

## Cleanup

I believe gmd_egress.{h, cpp} are currently there just to give you some context about how to implement the translation to EAIRoomObject, you may get rid of these once you are done.
