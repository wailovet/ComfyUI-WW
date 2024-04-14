import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const my_ui = {
  name: "ww.tools",
  setup() {},
  init: async () => {
    console.log("ww Registering UI extension");
  },

  /**
   * @param {import("./types/comfy").NodeType} nodeType
   * @param {import("./types/comfy").NodeDef} nodeData
   * @param {import("./types/comfy").App} app
   */
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // console.log(`Registering node ${nodeData.name}`);
    switch (nodeData.name) {
      case "WW_AccumulationPreviewImages":
        nodeType.prototype.onNodeCreated = function () {
          this.addWidget("button", `CleanAllPreview`, "cleanAllPreview", () => {
            api
              .fetchApi(`/extention/clean_all_preview`, {
                method: "POST",
                body: JSON.stringify({
                  clean_all_preview: true,
                }),
              })
              .then((resp) => {
                console.log(resp);
                alert("清理成功");
              });
          });
        };

      case "WW_PreviewTextNode":
        // Node Created
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          const ret = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined;

          let PreviewTextNode = app.graph._nodes.filter(
              (wi) => wi.type == nodeData.name
            ),
            nodeName = `${nodeData.name}_${PreviewTextNode.length}`;

          console.log(`Create ${nodeData.name}: ${nodeName}`);

          const wi = ComfyWidgets.STRING(
            this,
            nodeName,
            [
              "STRING",
              {
                default: "",
                placeholder: "Text message output...",
                multiline: true,
              },
            ],
            app
          );
          wi.widget.inputEl.readOnly = true;
          return ret;
        };
        // Function set value
        const outSet = function (texts) {
          if (texts.length > 0) {
            let widget_id = this?.widgets.findIndex(
              (w) => w.type == "customtext"
            );

            if (Array.isArray(texts))
              texts = texts
                .filter((word) => word.trim() !== "")
                .map((word) => word.trim())
                .join(" ");

            this.widgets[widget_id].value = texts;
            app.graph.setDirtyCanvas(true);
          }
        };

        // onExecuted
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (texts) {
          onExecuted?.apply(this, arguments);
          outSet.call(this, texts?.string);
        };
        // onConfigure
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (w) {
          onConfigure?.apply(this, arguments);
          if (w?.widgets_values?.length) {
            outSet.call(this, w.widgets_values);
          }
        };
    }
  },
};

app.registerExtension(my_ui);
