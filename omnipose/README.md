<img src="logo3.png" width="200" title="bacteria" alt="bacteria" align="right" vspace = "0">
<img src="logo.png" width="200" title="omnipose" alt="omnipose" align="center" vspace = "0">

### Why use Omnipose
Omnipose solves the over-segmentation problems of Cellpose on long, ansiotropic cells. This is particularly relevant for bacterial cells, but Omnipose is suitable for aribtrary cell shapes. 

### How to use Omnipose
To use Omnipose on bacterial cells, use `model_type=bact_omni`. For other cell types, try `model_type=cyto2_omni`. You can also use your own Cellpose models with `omni=True` to help solve problems with over-segmentatuon. 

### More about Omnipose
Omnipose builds on Cellpose in a number of ways described in our [paper](http://biorxiv.org/content/early/2021/11/04/2021.11.03.467199). It turns out that cell 'centers' are not well-defined, and this is a huge problem for any cells that are long and curved - generally leading to over-segmentation.

To fix this, we turned to the distance field (also known as the distance transform). This morphology-independent field replaces the uniform cell probability field of Cellpose, and its gradient replaces the heat-derived flow field of Cellpose. 

The distance field works the same way as cell probability output, meaning it can be thresholded (with the same values!) to seed cell masks. However, it carries a lot more information and we use it to refine the mask recontruction parameters when cells get too thin.

The flow field points towards the skeleton of the cell, and we built a new mask recontruction algorithm to leverage this fact. While cellpose tends to over-segement long cells, Omnipose can segment cells of any morphology whatsoever.

Omnipose as a fourth ouput class (as opposed to three in Cellpose). This extra output is a boundary probability field, which we found helped the network to make better predictions at cell boundaries. It is also useful in mask recontruction. 

### Licensing
Omnipose is completely free for non-commercial use. The code associated with Omnipose is dual-licensed under the
[GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html). For commercial use, contact uwcomotion@uw.edu. 
