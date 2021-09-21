import Game from "./src/game.js";
import View from "./src/viewWebGPU.js";
import Controller from "./src/controller.js";

const root = document.querySelector("#root");

const game = new Game();
const view = new View(root, 480, 520, 20, 10);
const controller = new Controller(game, view, view);

window.game = game;
window.view = view;
window.controller = controller;

console.log(game);
