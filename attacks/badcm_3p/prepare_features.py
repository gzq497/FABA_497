"""Reuse the project's bounded-pattern/direct-text BadCM adaptation."""
from attacks.three_path_prepare import main as staged_main


def build_generator(args, partition):
    from attacks.badcm.prepare_features import load_pattern, poison_raw_image, poison_raw_text
    from pathlib import Path
    pattern = load_pattern(Path(args.badcm_pattern_path))

    def generate(key, image, text, has_i, has_t, rng):
        trace = {'adaptation': 'bounded_pattern_direct_text'}
        if has_i:
            image = poison_raw_image(image, pattern, args.badcm_pixel_eps)
        if has_t:
            text, trace['text'] = poison_raw_text(text, args.badcm_text_trigger, key)
        return image, text, trace

    return generate, dict(source='Project BadCM adaptation: bounded image pattern + direct text trigger',
        badcm_pixel_eps=args.badcm_pixel_eps, badcm_text_trigger=args.badcm_text_trigger,
        malicious_selection='q_stratified_available_clients')


def main(dataset='crisis-mmd'):
    staged_main('badcm_3p', dataset)


if __name__ == '__main__':
    main()
