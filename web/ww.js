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
    setup() { },
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
                        api.fetchApi(
                            `/extention/clean_all_preview`,
                            {
                                method: "POST",
                                body: JSON.stringify({
                                    "clean_all_preview": true,
                                }),
                            }
                        ).then((resp) => {
                            console.log(resp);
                            alert("清理成功");
                        });

                    });
                }
        }
    },
};

app.registerExtension(my_ui);