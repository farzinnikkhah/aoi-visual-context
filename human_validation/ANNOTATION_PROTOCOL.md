# Human validation of visual grounding

## Assignment

For each provided video frame, independently identify and annotate all visible objects that belong to the 14 categories below. Annotate the complete scene, not only the object near a gaze or fixation point.

For every relevant object:

- Draw a tight box around the visible region.
- Assign one of the 14 categories.

Do not use or view automated model predictions. Do not look at the other annotator's boxes. Both sets of annotations will be compared for human agreement, then resolved into ground truth for evaluating the grounding model.

Work in AnyLabeling or LabelMe. The pilot set is 24 frames (3 in each of 8 conditions). Finish the pilot and the review meeting before annotating any further frames.

Images are 1920×1080. There are no detection overlays in the annotation package.

## Categories

| ID | Name | What to box |
|---|---|---|
| 1 | Moving vehicle | A car, truck, bus, or van in a travel lane, including one stopped in traffic |
| 2 | Parked vehicle | A vehicle in a parking position or pulled fully out of the travel lane at the curb |
| 3 | Sidewalk pavement | The walking surface, including the curb as its edge |
| 4 | Traffic signs | Stop, yield, speed, street-name, and other road signs meant for drivers |
| 5 | Crosswalk | The painted crossing on the roadway |
| 6 | Crossing signal | A pedestrian signal (walk / don't-walk / hand / person symbol) |
| 7 | General light features | Vehicle traffic lights, street lights, lamp posts, and light poles |
| 8 | Pedestrians | A person on foot, walking or standing |
| 9 | Cyclists | A person riding a bicycle |
| 10 | Storefront | A shop front: entrance, display window, and the shop's own sign |
| 11 | Buildings | A building facade that is not itself the shop front |
| 12 | Art and banners | A banner, flag, poster, mural, or advertisement |
| 13 | Natural elements | A tree, bush, hedge, or grass patch |
| 14 | Miscellaneous infrastructure | Bench, trash can, hydrant, parking meter, utility box, bollard, cone, fence, railing, bus stop, outdoor seating, awning, or a pole that is not already part of a sign or light |

## How to draw the box

- Touch the visible pixels of the object. Do not extend the box through an occluder to guess the hidden full shape.
- One object, one class. If two classes both fit, choose the better one.
- Annotate an object when about 20% or more of it is visible and you can identify the class. If you would be guessing, skip it.
- Skip a thing whose visible region is under 16 pixels on its longer side.
- Ignore the gaze point. It is not marked, and it does not decide what to annotate.

**One box per object:** classes 1, 2, 4, 6, 7, 8, 9, 12, and 14.

**One box per contiguous region:**

- Sidewalk: the continuous walking surface, not each slab and not a separate curb box.
- Crosswalk: the whole visible stripe group for that crossing, not each stripe. A second crossing direction gets its own box.
- Storefront: each shop front.
- Building: each facade. Do not box each window or door. Windows and doors belong to the building, or to the storefront when they are part of the shop.
- Vegetation: each tree, or one box around a continuous hedge or grass patch.

## Edge cases

**Moving versus parked.** Decide from position, not from imagined speed. In a travel lane, including stopped at a light or in a queue, is class 1. At the curb or in a parking space, out of the travel lane, is class 2. A vehicle queued in the lane is not parked. If the still is not enough, pick the better class.

**People.** A person on foot is class 8, including someone standing still. A person inside a vehicle is not a pedestrian. A person riding a bicycle is class 9 only. A bicycle with no rider is not annotated. A person walking a bicycle is class 8; do not also box the bicycle as a cyclist.

**Signs, signals, and lights.** A road sign is class 4. A pedestrian signal head is class 6. A vehicle traffic light or street light is class 7. If one pole carries both a traffic light and a pedestrian signal, draw two boxes. A shop's name sign is part of the storefront (class 10), not a traffic sign. An advertisement banner is class 12.

**Poles.** If the pole is the support for a sign, signal, or light you have already boxed, do not also box it as class 14. A bare pole with none of those is class 14.

**Shop and building.** When a ground-floor shop and the upper facade are both visible, box the shop front as class 10 and the upper facade as class 11. A plain wall or house with no shop is class 11 only.

**Road versus sidewalk.** The sidewalk box stays on the walking surface. The crosswalk box stays on the painted markings in the road. Do not cover the whole intersection with either box.

**Distant clutter.** Skip specks you cannot identify. For sidewalk, building, storefront, and vegetation, also skip a region whose longer side would be under about 48 pixels.

## What not to do

- Do not open model outputs, previous detection figures, or the other annotator's task.
- Do not annotate only the area you think someone looked at.
- Do not draw one box around a group of separate objects (two pedestrians are two boxes; a row of parked cars is one box per car).
- Do not add categories outside this list.

## What you deliver

Each box needs:

| Field | Values |
|---|---|
| Frame ID | The image name, without guessing a new id |
| Annotator | A1 or A2, as assigned |
| Object class | One of the 14 names |
| Bounding box | Drawn in AnyLabeling or LabelMe and exported as coordinates |

Use the 14 class names as the labels. Export the finished pilot and send that file. Do not calculate agreement or model scores.
