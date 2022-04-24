from logging import getLogger

logger = getLogger(__name__)


def load_best_model_weight(model, io_channel):
    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path, io_channel)


def save_as_best_model(model, io_channel, upload=False):
    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path, io_channel, upload)


def reload_best_model_weight_if_changed(model, io_channel):
    if model.config.model.distributed:
        return load_best_model_weight(model, io_channel)
    else:
        logger.debug('start reload the best model if changed')
        digest = model.fetch_digest(model.config.resource.model_best_weight_path)
        if digest != model.digest:
            return load_best_model_weight(model, io_channel)

        logger.debug('the best model is not changed')
        return False
