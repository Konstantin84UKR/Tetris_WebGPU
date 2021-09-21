export default class Controller {
  constructor(game, view, viewWebGPU) {
    this.game = game;
    this.view = view;
    this.viewWebGPU = viewWebGPU;
    this.isPlaying = false;
    this.gameLoopID = null;
    this.intervalID = null;

    document.addEventListener("keydown", this.handleKeyDown.bind(this));
    this.play();
  }

  update() {
    game.movePieceDown();
    this.updateView();
  }

  play() {
    this.isPlaying = true;
    this.startTimer();
    this.updateView();
    this.gameLoop(this);
  }

  pause() {
    this.isPlaying = false;
    this.stopTimer();
    this.updateView();
    this.view.renderPauseScreen();
  }

  startTimer() {
    const speed = 1000 - this.game.getState().level * 100;

    if (!this.intervalID) {
      this.intervalID = setInterval(
        () => {
          this.update();
        },
        speed > 0 ? speed : 100
      );
    }
  }

  stopTimer() {
    if (this.intervalID) {
      clearInterval(this.intervalID);
      this.intervalID = null;
    }
  }

  updateView() {
    const state = this.game.getState();

    if (state.isGameOwer) {
      this.view.renderEndScreen(state);
      this.isPlaying = false;
    } else if (!this.isPlaying) {
      this.view.renderPauseScreen(state);
    } else {
      this.view.renderMainScreen(state);
    }
  }

  reset() {
    this.game.reset();
    this.play();
  }

  handleKeyDown(event) {
    const state = this.game.getState();

    switch (event.keyCode) {
      case 13: // ENTER
        if (state.isGameOwer) {
          this.reset();
        } else if (this.isPlaying) {
          this.pause();
        } else {
          this.play();
        }

        break;
      case 37:
        this.game.movePieceLeft();
        this.updateView();
        break;
      case 38:
        this.game.rotatePiece();
        this.updateView();
        break;
      case 39:
        this.game.movePieceRight();
        this.updateView();
        break;
      case 40:
        this.game.movePieceDown();
        this.updateView();
        break;
    }
  }

  gameLoop(thisRL) {
    let dt = 0;
    let old_time = 0;

    const animate = function (time) {
      if (!old_time) old_time = time;
      if (Math.abs(time - old_time) >= 1000 / 30) {
        const stateForWebGPU = thisRL.game.getState();
        thisRL.viewWebGPU.state = stateForWebGPU;
        old_time = time;
      }
      if (thisRL.isPlaying == false) {
        return 0;
      }
      window.requestAnimationFrame(animate);
    };
    animate(0);
  }
}
